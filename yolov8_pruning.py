# This code is adapted from Issue [#147](https://github.com/VainF/Torch-Pruning/issues/147), implemented by @Hyunseok-Kim0.
import argparse
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.engine.model import TASK_MAP
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import initialize_weights, de_parallel

import torch_pruning as tp


def save_pruning_performance_graph(x, y1, y2, y3):
    """
    프루닝 진행 단계별 성능 변화를 이중 y축 그래프로 그려 pruning_perf_change.png로 저장.

    x  : 각 스텝의 프루닝 비율 (x축, 오른쪽→왼쪽으로 반전하여 프루닝 진행 방향 표시)
    y1 : 파인튜닝 후 mAP (recovered mAP) — 왼쪽 y축
    y2 : MACs — 오른쪽 y축 (y2[0] 대비 비율로 정규화)
    y3 : 프루닝 직후 mAP, 파인튜닝 전 (pruned mAP) — 왼쪽 y축
    """
    try:
        plt.style.use("ggplot")
    except:
        pass

    x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
    y2_ratio = y2 / y2[0]

    # create the figure and the axis object
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP')
    ax.plot(x, y1, label='recovered mAP')
    ax.scatter(x, y1)
    ax.plot(x, y3, color='tab:gray', label='pruned mAP')
    ax.scatter(x, y3, color='tab:gray')

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('MACs')
    ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
    ax2.scatter(x, y2_ratio, color='tab:orange')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(105, -5)
    ax.set_ylim(0, max(y1) + 0.05)
    ax2.set_ylim(0.05, 1.05)

    # calculate the highest and lowest points for each set of data
    max_y1_idx = np.argmax(y1)
    min_y1_idx = np.argmin(y1)
    max_y2_idx = np.argmax(y2)
    min_y2_idx = np.argmin(y2)
    max_y1 = y1[max_y1_idx]
    min_y1 = y1[min_y1_idx]
    max_y2 = y2_ratio[max_y2_idx]
    min_y2 = y2_ratio[min_y2_idx]

    # add text for the highest and lowest values near the points
    ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
    ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
    ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
    ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

    plt.title('Comparison of mAP and MACs with Pruning Ratio')
    plt.savefig('pruning_perf_change.png')


def infer_shortcut(bottleneck):
    '''
    YOLOv8 Bottleneck은 ResNet처럼 입력을 출력에 더하는 skip connection이 있을 수도 있어:
    x ──→ cv1 ──→ cv2 ──→ (+) ──→ output
    │                      ↑
    └──────────────────────┘  ← shortcut (add=True일 때만)
    add=True 조건: shortcut 파라미터가 True 이고 c1 == c2 (채널 수가 같아야 덧셈 가능)

    왜 하는가?
    replace_c2f_with_c2f_v2에서 C2f_v2를 새로 만들 때 shortcut 인자가 필요해:

    '''
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


class C2f_v2(nn.Module):
    # torch-pruning 호환을 위해 C2f를 재구성한 버전
    # 원래 C2f: cv1 하나로 2c 채널 출력 후 chunk(2,1)로 분리 → pruning 의존성 추적 불가
    # C2f_v2: cv0, cv1 두 개의 독립 conv로 명시적 분리 → pruning이 각 채널을 독립적으로 추적 가능
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)  # 원래 C2f.cv1의 앞 절반 (chunk 앞부분)
        self.cv1 = Conv(c1, self.c, 1, 1)  # 원래 C2f.cv1의 뒤 절반 (chunk 뒷부분)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    # cv2(출력 conv)와 m(Bottleneck 리스트)은 구조 동일 → 직접 참조 복사
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # C2f.cv1 weight shape: [2c, C_in, 1, 1]
    # dim=0(출력 채널)으로 반 잘라 cv0, cv1에 각각 할당
    # → 런타임에 chunk(2,1)로 feature map을 쪼개던 것과 동일한 효과를 가중치 레벨에서 미리 분리
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Conv 클래스는 conv + bn + act 세트이므로 BN 파라미터도 동일하게 반씩 분배
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # cv1 이외의 나머지 가중치(cv2, m 등)는 그대로 복사
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # self.c 등 숫자형 attribute 복사
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    # 모델 트리를 재귀 탐색하여 C2f를 모두 C2f_v2로 교체
    # named_children()은 직계 자식만 반환 → leaf 모듈(Conv 등)은 자식이 없어 루프 즉시 종료 (재귀 자연 종료)
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # 기존 C2f에서 하이퍼파라미터를 읽어 동일한 구조의 C2f_v2 생성 후 가중치 이식
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)


def save_model_v2(self: BaseTrainer):
    """
    프루닝 모드 전용 모델 저장 함수. ultralytics/yolo/engine/trainer.py의 save_model을 대체.

    원본과 달리 float16(half precision) 변환을 하지 않는다.
    프루닝으로 구조가 바뀐 모델(C2f_v2 등)을 float16으로 변환하면 state_dict 불일치가 생길 수 있기 때문.
    train_v2에서 pruning=True일 때 self.trainer.save_model을 이 함수로 교체한다.
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt


def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/yolo/engine/trainer.py
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    학습 완료된 .pt 파일에서 불필요한 데이터를 제거해 추론 전용 경량 파일로 만든다.
    ultralytics/yolo/utils/torch_utils.py의 strip_optimizer를 대체.

    - EMA가 있으면 model을 EMA 가중치로 교체 (더 정확한 가중치)
    - optimizer / ema / updates 키를 None으로 제거 → 파일 크기 감소
    - 모든 파라미터에 requires_grad=False 설정 (추론 전용)
    - 원본과 달리 float16 변환을 하지 않는다 (프루닝된 커스텀 모듈 보호)
    - final_eval_v2에서 학습 종료 시 last.pt, best.pt에 대해 자동 호출됨
    """
    x = torch.load(f, map_location=torch.device('cpu'))
    args = {**DEFAULT_CFG_DICT, **x['train_args']}  # combine model args with default args, preferring model args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def train_v2(self: YOLO, pruning=False, **kwargs):
    """
    Disabled loading new model when pruning flag is set. originated from ultralytics/yolo/engine/model.py
    """

    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)

    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model

        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)


def prune(args):
    # =========================================================================
    # [1단계] 초기화
    # =========================================================================

    # 사전학습된 YOLO 모델 로드 (.pt 체크포인트)
    model = YOLO(args.model)

    # train_v2를 YOLO 인스턴스의 바운드 메서드로 monkey-patch
    # 이유: 원본 YOLO.train()은 호출마다 새 모델 가중치를 로드하는데,
    #       프루닝 후에는 레이어 구조 자체가 바뀌어 원본 가중치를 덮어씌우면 안 됨.
    #       train_v2는 pruning=True 시 이 초기화 로직을 건너뜀.
    model.__setattr__("train_v2", train_v2.__get__(model))

    pruning_cfg = yaml_load(check_yaml(args.cfg))

    # 파인튜닝 시 쓸 배치 크기를 미리 저장 (검증 단계에선 1로 덮어쓰기 때문)
    batch_size = pruning_cfg['batch']

    # 샘플 코드용 고정값. 실제 사용 시 cfg 파일에서 설정하는 것을 권장.
    pruning_cfg['data'] = "coco128.yaml"
    pruning_cfg['epochs'] = 10

    # BN을 train 모드로 전환해야 running_mean/var가 업데이트됨
    model.model.train()

    # C2f → C2f_v2 교체
    # 이유: torch-pruning은 정적 계산 그래프 추적으로 채널 의존성을 파악하는데,
    #       원본 C2f의 chunk(2,1) 연산은 추적 불가.
    #       C2f_v2는 이를 cv0, cv1 두 Conv로 명시적으로 분리해 추적 가능하게 함.
    replace_c2f_with_c2f_v2(model.model)

    # BN의 eps·momentum 재설정, ReLU inplace 활성화 등 초기 상태 정규화
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace

    # 모든 파라미터의 gradient 활성화 (프루닝·파인튜닝에 필요)
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    # MACs/파라미터 수 측정을 위한 더미 입력 (실제 추론은 하지 않음)
    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)

    # 각 프루닝 스텝의 성능 지표를 기록하는 리스트
    macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []

    # 프루닝 전 기준 MACs·파라미터 수 측정 (나중에 speed-up 비율 계산에 사용)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    # =========================================================================
    # [2단계] 프루닝 전 기준선(baseline) 측정
    # =========================================================================

    pruning_cfg['name'] = f"baseline_val"
    pruning_cfg['batch'] = 1  # 검증은 배치 1로 진행
    validation_model = deepcopy(model)
    metric = validation_model.val(**pruning_cfg)
    init_map = metric.box.map  # 프루닝 전 초기 mAP (조기 종료 판단 기준)

    # 기준선을 리스트 인덱스 0에 기록
    macs_list.append(base_macs)
    nparams_list.append(100)        # 프루닝 전 = 파라미터 100% 상태
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")

    # =========================================================================
    # [3단계] 스텝당 프루닝 비율 계산
    # =========================================================================

    # 목표: N 스텝 후 전체 파라미터의 target_prune_rate 만큼 제거
    # 유도: (1 - r)^N = (1 - target_rate)  →  r = 1 - (1-target_rate)^(1/N)
    # 예)  target=50%, steps=16  →  r ≈ 4.3% (매 스텝 동일 비율 제거)
    pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)

    # =========================================================================
    # [4단계] 반복 프루닝 루프
    #   각 스텝: 프루닝 → pre-val → 파인튜닝 → post-val → 조기 종료 판단
    # =========================================================================

    for i in range(args.iterative_steps):

        # 이전 스텝의 pruner.step()이 일부 gradient를 비활성화할 수 있으므로 재활성화
        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        # ------------------------------------------------------------------
        # 4-A. 프루닝 제외 레이어 지정
        # ------------------------------------------------------------------
        ignored_layers = []
        unwrapped_parameters = []
        for m in model.model.modules():
            if isinstance(m, (Detect,)):
                # Detect 헤드는 출력 채널이 (앵커 수 × 클래스 수)에 고정되어 있어
                # 채널 수를 변경하면 후처리(디코딩) 단계가 깨짐 → 프루닝 제외
                ignored_layers.append(m)

        # ------------------------------------------------------------------
        # 4-B. GroupNormPruner 구성
        # ------------------------------------------------------------------
        example_inputs = example_inputs.to(model.device)
        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,          # 계산 그래프 추적용 더미 입력
            importance=tp.importance.GroupMagnitudeImportance(),  # L2 norm 기준 중요도: 값이 작은 필터 그룹부터 제거
            iterative_steps=1,       # pruner.step() 한 번에 전체 프루닝 수행
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )

        # Test regularization
        #output = model.model(example_inputs)
        #(output[0].sum() + sum([o.sum() for o in output[1]])).backward()
        #pruner.regularize(model.model)

        # ------------------------------------------------------------------
        # 4-C. 실제 프루닝 실행
        # ------------------------------------------------------------------
        tp.utils.print_tool.before_pruning(model.model)  # 프루닝 전 레이어별 채널 수 출력
        pruner.step()                                     # 중요도 낮은 필터 제거 + 의존 레이어(BN, 다음 Conv 등) 자동 조정
        tp.utils.print_tool.after_pruning(model.model, do_print=True)  # 프루닝 후 변경된 채널 수 출력

        # ------------------------------------------------------------------
        # 4-D. 파인튜닝 전 검증 (프루닝만으로 인한 mAP 드롭 확인)
        # ------------------------------------------------------------------
        pruning_cfg['name'] = f"step_{i}_pre_val"
        pruning_cfg['batch'] = 1
        validation_model.model = deepcopy(model.model)
        metric = validation_model.val(**pruning_cfg)
        pruned_map = metric.box.map  # 파인튜닝 전 mAP

        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs.to(model.device))
        current_speed_up = float(macs_list[0]) / pruned_macs  # 기준 MACs 대비 현재 MACs 비율
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
              f"mAP={pruned_map}, speed up={current_speed_up}")

        # ------------------------------------------------------------------
        # 4-E. 파인튜닝 (프루닝으로 손실된 정확도 회복)
        # ------------------------------------------------------------------
        # pruner.step() 이후 gradient가 비활성화될 수 있으므로 재활성화
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = f"step_{i}_finetune"
        pruning_cfg['batch'] = batch_size  # 검증에서 1로 바꿨던 배치 크기 복원
        # pruning=True: 모델 가중치를 새로 로드하지 않고 현재 (프루닝된) 모델로 학습
        model.train_v2(pruning=True, **pruning_cfg)

        # ------------------------------------------------------------------
        # 4-F. 파인튜닝 후 검증 (mAP 회복 확인)
        # ------------------------------------------------------------------
        pruning_cfg['name'] = f"step_{i}_post_val"
        pruning_cfg['batch'] = 1
        # 파인튜닝 중 저장된 best 체크포인트를 로드해 검증
        validation_model = YOLO(model.trainer.best)
        metric = validation_model.val(**pruning_cfg)
        current_map = metric.box.map
        print(f"After fine tuning mAP={current_map}")

        # 이번 스텝 지표 기록
        macs_list.append(pruned_macs)
        nparams_list.append(pruned_nparams / base_nparams * 100)  # 남은 파라미터 비율(%)
        pruned_map_list.append(pruned_map)   # 파인튜닝 전 mAP
        map_list.append(current_map)         # 파인튜닝 후 mAP

        # pruner 객체는 계산 그래프 참조를 내부에 보관하므로 다음 스텝 전에 명시적으로 해제
        del pruner

        # 프루닝 비율 vs mAP/MACs 변화 그래프 갱신
        save_pruning_performance_graph(nparams_list, map_list, macs_list, pruned_map_list)

        # ------------------------------------------------------------------
        # 4-G. 조기 종료: 허용 mAP 드롭을 초과하면 중단
        # ------------------------------------------------------------------
        if init_map - current_map > args.max_map_drop:
            print("Pruning early stop")
            break

    # 최종 프루닝 모델을 ONNX로 내보내기 (RKNN 변환 등 후속 작업에 사용)
    model.export(format='onnx')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8m.pt', help='Pretrained pruning target model file')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--iterative-steps', default=16, type=int, help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.5, type=float, help='Target pruning rate')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')

    args = parser.parse_args()

    prune(args)
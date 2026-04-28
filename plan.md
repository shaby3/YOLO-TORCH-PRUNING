# Plan: yolov8_pruning.py → prune.py 이식

## Context

현재 `prune.py`는 단순히 `model.train(prune=True)`를 호출하며, 실제 pruner는 `trainer._setup_train()`
안에서 한 번만 실행된다. `iterative_steps > 1`은 사실상 동작하지 않고, legacy C2f weight 변환도 없다.

목표: `yolov8_pruning.py`의 iterative pruning 전략(prune → fine-tune → repeat)을 `prune.py`에
이식하고, pretrained 체크포인트 호환을 위한 C2f weight transfer를 추가한다.

**수정 파일: `prune.py` 하나만.** ultralytics 내부 코드 수정 없음.

---

## 구조 (5 섹션)

### Section 1 — Imports

```python
import math
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.block import C2f, Attention
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import RANK
import torch_pruning as tp
```

---

### Section 2 — C2f Weight Transfer

**`_has_legacy_c2f(pt_path: str) -> bool`**
- `torch.load(pt_path)` 로 raw checkpoint 로드
- `ckpt.get('ema') or ckpt.get('model')` 에서 state_dict 추출
- `any('cv0' in k for k in state)` 가 False이면 legacy (cv1이 2c 출력인 구조)

**`transfer_pretrained_c2f(model: YOLO, pt_path: str) -> None`**
- `_has_legacy_c2f`가 False면 early return (no-op)
- old_state에서 새 모델의 new_state로 복사:
  1. shape 동일한 키는 직접 복사
  2. `cv1.conv.weight` 키 중 `shape[0] == 2 * new_c` 인 것 → dim=0 기준 절반 분리:
     - `cv0.conv.weight = old_w[:half]`
     - `cv1.conv.weight = old_w[half:]`
  3. BN 파라미터 4종(`weight`, `bias`, `running_mean`, `running_var`)도 동일하게 분리
- `model.model.load_state_dict(new_state, strict=False)`

---

### Section 3 — Fine-tune Helper

**`_train_pruned(yolo: YOLO, **kwargs) -> None`**

`yolov8_pruning.py`의 `train_v2(pruning=True)`에 해당. Pruning 후 구조가 바뀐 모델을
디스크에서 재로드하지 않고 trainer에 직접 전달한다.

```python
def _train_pruned(yolo: YOLO, **kwargs) -> None:
    overrides = yolo.overrides.copy()
    overrides.update(kwargs)
    overrides['mode'] = 'train'

    trainer_cls = yolo.task_map[yolo.task]['trainer']
    trainer = trainer_cls(overrides=overrides, _callbacks=yolo.callbacks)
    trainer.model = yolo.model        # setup_model() 우회 — 현재 pruned model 직접 주입
    trainer.hub_session = getattr(yolo, 'session', None)
    trainer.prune = False             # fine-tune만, 추가 pruning 없음
    trainer.train()

    if RANK in {-1, 0}:
        ckpt_path = trainer.best if trainer.best.exists() else trainer.last
        yolo.model, _ = attempt_load_one_weight(str(ckpt_path))
        yolo.model = yolo.model.float()   # fp16 저장 → fp32 복원 (다음 pruning step용)
        yolo.overrides = yolo.model.args
    yolo.trainer = trainer
```

핵심: `trainer.model = yolo.model` 설정으로 trainer의 `setup_model()`이 yaml/pt에서
모델을 다시 로드하는 것을 막는다. (ultralytics trainer는 `self.model`이 `nn.Module`이면
로드를 건너뜀)

---

### Section 4 — Iterative Pruning Loop

**`iterative_prune(model, data, finetune_epochs, target_prune_ratio, iterative_steps=1, imgsz=640, batch=8, device=None, name='prune', max_map_drop=0.10)`**

```
per_step_ratio = 1 - (1 - target_prune_ratio)^(1 / iterative_steps)
```
→ N 스텝 후 누적 pruning이 정확히 target_prune_ratio에 수렴

루프 (step = 0 .. iterative_steps-1):
1. `model.model.float().train()` + requires_grad 재활성화
2. `ignored_layers` 구성: `Detect` + `Attention` 인스턴스 수집
3. `MagnitudePruner` 생성:
   ```python
   tp.pruner.MagnitudePruner(
       model.model, example_inputs,
       importance=tp.importance.MagnitudeImportance(p=2),
       iterative_steps=1,          # 외부 루프가 iterative 관리
       pruning_ratio=per_step_ratio,
       ignored_layers=ignored_layers,
   )
   ```
4. `pruner.step()` → MACs/Params 출력
5. `del pruner` (계산 그래프 참조 해제 — 다음 step 전 필수)
6. requires_grad 재활성화 (`pruner.step()` 이후 일부 비활성화될 수 있음)
7. `_train_pruned(model, name=f'{name}_step{step}', data=data, epochs=finetune_epochs, ...)`
8. mAP 검증
9. `init_map - current_map > max_map_drop` 이면 early stop

`iterative_steps=1`인 경우: baseline val + early stop 로직은 동작하지만 루프가 1회만
실행 → 기존 동작과 동일.

---

### Section 5 — Entry Point

**`prunetrain(model, train_epochs, prune_epochs=0, quick_pruning=True, prune_ratio=0.5, prune_iterative_steps=1, data='coco.yaml', name='yolo11', imgsz=640, batch=8, device=None, sparse_training=False, max_map_drop=0.10)`**

```python
def prunetrain(model: YOLO, ...):
    finetune_epochs = prune_epochs if not quick_pruning else train_epochs

    if not quick_pruning:
        assert train_epochs > 0 and prune_epochs > 0
        model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch,
                    device=device, name=name, prune=False,
                    sparse_training=sparse_training)

    iterative_prune(
        model, data=data,
        finetune_epochs=finetune_epochs,
        target_prune_ratio=prune_ratio,
        iterative_steps=prune_iterative_steps,
        imgsz=imgsz, batch=batch, device=device,
        name=name, max_map_drop=max_map_drop,
    )
```

사용 예시:
```python
model = YOLO('yolo11.yaml')

# legacy pretrained 체크포인트 사용 시:
# model = YOLO('pretrained.pt')
# transfer_pretrained_c2f(model, 'pretrained.pt')

prunetrain(
    model,
    quick_pruning=False,
    data='coco.yaml',
    train_epochs=10,
    prune_epochs=10,
    imgsz=640, batch=8, device=[0],
    name='yolo11',
    prune_ratio=0.5,
    prune_iterative_steps=1,
    sparse_training=False,
    max_map_drop=0.10,
)
```

---

## 수정 파일 목록

| 파일 | 변경 내용 |
|---|---|
| `prune.py` | 전체 재작성 (5섹션) |

수정 없음: `ultralytics/engine/trainer.py`, `ultralytics/engine/model.py`, `ultralytics/nn/modules/block.py`

---

## 중요 설계 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| Pruner 생성 위치 | `prune.py` (외부 루프) | trainer의 pruner와 분리; step간 fine-tune 가능 |
| Fine-tune 방식 | `_train_pruned` (trainer.model 직접 주입) | model reload 방지; `train_v2`와 동일 원리 |
| Importance | `MagnitudeImportance(p=2)` | 요구사항; 프로젝트 기존 설정 유지 |
| fp16 → fp32 복원 | `_train_pruned` 내 `.float()` | save_model이 `.half()` 저장 → 다음 step pruner가 fp32 필요 |
| `iterative_steps=1` 시 | loop 1회 실행 → 기존 동작 동일 | 하위 호환 |
| ignored_layers | `Detect` + `Attention` | 기존 프로젝트 설정 유지 |

---

## 검증 방법

1. **기본 동작**: `python prune.py` 실행 → 오류 없이 학습 루프 진행 확인
2. **transfer_pretrained_c2f**: legacy `.pt` 로드 후 C2f의 `cv0.conv.weight`가 old `cv1.conv.weight[:half]`와 일치하는지 확인
3. **iterative_steps=2**: 2회 prune+finetune 사이클 실행 확인, 각 step 후 MACs 감소 확인
4. **early stop**: `max_map_drop=0.0` 설정 → 1 step 후 중단 확인
5. **`_train_pruned` model 유지**: fine-tune 전후 `model.model` 구조(채널 수) 동일 확인

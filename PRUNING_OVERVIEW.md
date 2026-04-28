# Pruning 구현 정리

프로젝트의 pruning 코드는 `prune.py` + `ultralytics/` 로 구성된다.
`yolov8_pruning.py`는 torch-pruning examples에서 가져온 참고용 파일이며 실제 프로젝트 플로우와 무관하다.

---

## 프로젝트 pruning 플로우

```
prune.py
  └─ model.train(prune=True, ...)
       └─ ultralytics/engine/model.py  (trainer에 prune 설정 주입)
            └─ ultralytics/engine/trainer.py  (MagnitudePruner 생성 및 실행)
```

---

## 1. `ultralytics/nn/modules/block.py` — C2f 수정

표준 ultralytics의 C2f는 `cv1(x).chunk(2, 1)`을 사용하는데,
이는 torch-pruning의 계산 그래프 추적을 방해한다.
프로젝트에서는 C2f를 이미 pruning에 적합한 구조로 수정해두었다.

```python
# 원본 (표준 ultralytics)
# y = list(self.cv1(x).chunk(2, 1))

# 수정본 (이 프로젝트)
self.cv0 = Conv(c1, self.c, 1, 1)
self.cv1 = Conv(c1, self.c, 1, 1)

def forward(self, x):
    y = [self.cv0(x), self.cv1(x)]   # chunk 대신 두 독립 Conv
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))
```

`forward_split()`도 존재하며, `cv0`가 없는 구 체크포인트와의 호환을 위해 fallback 분기를 포함한다.

---

## 2. `ultralytics/cfg/default.yaml` — Pruning 관련 설정값

```yaml
prune: False                # 프루닝 실행 여부
prune_ratio: 0.66874        # 제거할 채널 비율
prune_iterative_steps: 1    # pruner.step() 분할 횟수
prune_load: False           # 프루닝 후 원래 가중치 재로드 여부
sparse_training: False      # sparse regularization 적용 여부 (실험적)
```

---

## 3. `ultralytics/engine/model.py` — kwargs 주입

`YOLO.train()` 호출 시 pruning 관련 kwargs를 trainer에 직접 주입한다.

```python
# line 799–801
self.trainer.prune = kwargs.get("prune", False)
self.trainer.prune_ratio = kwargs.get("prune_ratio", 0.5)
self.trainer.prune_iterative_steps = kwargs.get("prune_iterative_steps", 1)
```

`save_model`, `final_eval`은 기본 구현을 그대로 사용한다.

---

## 4. `ultralytics/engine/trainer.py` — Pruner 생성 및 실행

**초기화** (`__init__`):
```python
self.prune = self.args.prune
self.prune_ratio = self.args.prune_ratio
self.prune_iterative_steps = self.args.prune_iterative_steps
self.prune_load = self.args.prune_load
self.sparse_training = self.args.sparse_training
self.pruner = None
```

**Pruner 생성** (`_setup_train()`):
```python
ignored_layers = []
for m in self.model.modules():
    if isinstance(m, (Detect, Attention)):   # 출력 구조 고정 레이어 제외
        ignored_layers.append(m)

self.pruner = tp.pruner.MagnitudePruner(
    self.model,
    example_inputs,
    importance=tp.importance.MagnitudeImportance(p=2),  # L2 norm 기준
    iterative_steps=self.prune_iterative_steps,
    pruning_ratio=self.prune_ratio,
    ignored_layers=ignored_layers,
)
```

**프루닝 실행** (동일 함수):
```python
if self.prune:
    self.pruner.step()
    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(...)

    if self.prune_load:   # 프루닝 후 원래 가중치 복원 (선택)
        weights, _ = attempt_load_one_weight(path)
        self.model.load(weights)
```

**Sparse training** (학습 루프):
```python
# 매 epoch 시작 시 (BN weight에 L1 regularization 준비)
if self.sparse_training:
    self.pruner.update_regularizer()

# loss.backward() 이후, optimizer.step() 이전
if self.sparse_training:
    self.pruner.regularize(self.model, self.loss)
```

sparse training을 먼저 돌려 중요도가 낮은 채널의 BN weight를 0으로 수렴시킨 뒤
pruning을 적용하면 accuracy drop을 줄일 수 있다.

---

## 5. `prune.py` — 사용자 진입점

```python
model = YOLO('yolo11.yaml')  # 또는 pretrained .pt

prunetrain(
    quick_pruning=False,        # False: 사전학습 후 프루닝
    data='coco.yaml',
    train_epochs=10,            # 프루닝 전 사전학습 epoch
    prune_epochs=10,            # 프루닝 후 파인튜닝 epoch
    prune_ratio=0.5,
    prune_iterative_steps=1,
    sparse_training=False,
)
```

**quick_pruning=False** (Normal Pruning):
1. `model.train(prune=False)` → 사전학습
2. `model.train(prune=True)` → 프루닝 + 파인튜닝

**quick_pruning=True** (Quick Pruning):
1. `model.train(prune=True)` → 사전학습 없이 바로 프루닝 + 파인튜닝

---

## 6. `Attention` — ignored_layers에 반드시 포함해야 하는 이유

`ultralytics/nn/modules/block.py` L1329의 `Attention` 클래스는 채널 pruning과 구조적으로 호환되지 않는다.

**이유 1: `.view()` / `.split()` 의존성**

```python
qkv = self.qkv(x)  # output channels = dim + 2 * key_dim * num_heads
q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
    [self.key_dim, self.key_dim, self.head_dim], dim=2
)
```

`qkv`의 출력 채널 수가 정확히 `num_heads * (key_dim * 2 + head_dim)`이어야 한다.
채널 1개라도 pruning되면 `.view()`에서 shape mismatch로 런타임 오류 발생.

**이유 2: `pe` depthwise conv**

```python
self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # groups = dim (고정)
```

`groups=dim`이므로 입력 채널이 정확히 `dim`이어야 한다.
상위 레이어 pruning으로 입력 채널이 바뀌어도 깨진다.

→ `ignored_layers`에서 빼면 안 됨.

---

## `yolov8_pruning.py` 비교 (참고용)

| 항목 | 프로젝트 (`prune.py` + `ultralytics/`) | `yolov8_pruning.py` |
|---|---|---|
| 출처 | 이 프로젝트 | torch-pruning examples |
| Pruner | `MagnitudePruner` | `GroupNormPruner` |
| Importance | `MagnitudeImportance(p=2)` | `GroupMagnitudeImportance()` |
| C2f 구조 | 프로젝트 C2f에 이미 반영 | 별도 `C2f_v2` 클래스로 교체 |
| ignored layers | `Detect`, `Attention` | `Detect`만 |
| float16 저장 | 기본 `save_model` (`.half()` 적용) | `save_model_v2` (비활성화) |
| step별 pre/post 검증 | 없음 | 있음 |
| 조기 종료 | 없음 | `max_map_drop` 기준 |
| Sparse training | 지원 | 없음 |
| ONNX export | 없음 | 있음 |

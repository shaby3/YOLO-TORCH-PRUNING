import math
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.block import C2f, Attention
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import RANK
import torch_pruning as tp


# Section 2 — C2f Weight Transfer

def _has_legacy_c2f(pt_path: str) -> bool:
    ckpt = torch.load(pt_path, map_location='cpu')
    state = (ckpt.get('ema') or ckpt.get('model')).state_dict()
    return not any('cv0' in k for k in state)


def transfer_pretrained_c2f(model: YOLO, pt_path: str) -> None:
    if not _has_legacy_c2f(pt_path):
        return
    ckpt = torch.load(pt_path, map_location='cpu')
    old_state = (ckpt.get('ema') or ckpt.get('model')).state_dict()
    new_state = model.model.state_dict()

    for k in list(new_state.keys()):
        if k not in old_state:
            continue
        old_w, new_w = old_state[k], new_state[k]
        if old_w.shape == new_w.shape:
            new_state[k] = old_w
        elif old_w.shape[0] == 2 * new_w.shape[0]:
            half = old_w.shape[0] // 2
            new_state[k] = old_w[half:]
            cv0_k = k.replace('.cv1.', '.cv0.')
            if cv0_k in new_state:
                new_state[cv0_k] = old_w[:half]

    model.model.load_state_dict(new_state, strict=False)


# Section 3 — Fine-tune Helper

def _train_pruned(yolo: YOLO, **kwargs) -> None:
    overrides = yolo.overrides.copy()
    overrides.update(kwargs)
    overrides['mode'] = 'train'

    trainer_cls = yolo.task_map[yolo.task]['trainer']
    trainer = trainer_cls(overrides=overrides, _callbacks=yolo.callbacks)
    trainer.model = yolo.model
    trainer.hub_session = getattr(yolo, 'session', None)
    trainer.prune = False
    trainer.train()

    if RANK in {-1, 0}:
        ckpt_path = trainer.best if trainer.best.exists() else trainer.last
        yolo.model, _ = attempt_load_one_weight(str(ckpt_path))
        yolo.model = yolo.model.float()
        yolo.overrides = yolo.model.args
    yolo.trainer = trainer


# Section 4 — Iterative Pruning Loop

def iterative_prune(model: YOLO, data, finetune_epochs, target_prune_ratio,
                    iterative_steps=1, imgsz=640, batch=8, device=None,
                    name='prune', max_map_drop=0.10):
    per_step_ratio = 1 - (1 - target_prune_ratio) ** (1 / iterative_steps)

    init_map = model.val(data=data, imgsz=imgsz, batch=batch, device=device).box.map

    for step in range(iterative_steps):
        model.model.float().train()
        for p in model.model.parameters():
            p.requires_grad_(True)

        ignored_layers = [m for m in model.model.modules()
                          if isinstance(m, (Detect, Attention))]

        example_inputs = torch.zeros(1, 3, imgsz, imgsz)
        pruner = tp.pruner.MagnitudePruner(
            model.model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=2),
            iterative_steps=1,
            pruning_ratio=per_step_ratio,
            ignored_layers=ignored_layers,
        )
        pruner.step()

        macs, params = tp.utils.count_ops_and_params(model.model, example_inputs)
        print(f'[Step {step}] MACs: {macs / 1e9:.2f}G, Params: {params / 1e6:.2f}M')

        del pruner

        for p in model.model.parameters():
            p.requires_grad_(True)

        _train_pruned(model, name=f'{name}_step{step}', data=data,
                      epochs=finetune_epochs, imgsz=imgsz, batch=batch, device=device)

        current_map = model.val(data=data, imgsz=imgsz, batch=batch, device=device).box.map
        print(f'[Step {step}] mAP: {current_map:.4f} (init: {init_map:.4f})')

        if init_map - current_map > max_map_drop:
            print(f'Early stop: mAP drop {init_map - current_map:.4f} > {max_map_drop}')
            break


# Section 5 — Entry Point

def prunetrain(model: YOLO, train_epochs, prune_epochs=0, quick_pruning=True,
               prune_ratio=0.5, prune_iterative_steps=1, data='coco.yaml',
               name='yolo11', imgsz=640, batch=8, device=None,
               sparse_training=False, max_map_drop=0.10):
    ckpt_path = getattr(model, 'ckpt_path', None)
    if ckpt_path and str(ckpt_path).endswith('.pt'):
        transfer_pretrained_c2f(model, str(ckpt_path))

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


model = YOLO('yolo11.yaml')
# model = YOLO('yolov8n.pt')  # .pt 로드 시 transfer_pretrained_c2f 자동 호출

# Normal Pruning
prunetrain(
    model,
    quick_pruning=False,       # Quick Pruning or not
    data='coco.yaml',          # Dataset config
    train_epochs=10,           # Epochs before pruning
    prune_epochs=10,           # Epochs after pruning
    imgsz=640,                 # Input size
    batch=8,                   # Batch size
    device=[0],                # GPU devices
    name='yolo11',             # Save name
    prune_ratio=0.5,           # Pruning Ratio (50%)
    prune_iterative_steps=1,   # Pruning Iterative Steps
    sparse_training=False,     # Experimental, Allow Sparse Training Before Pruning
)
# Quick Pruning (prune_epochs no need)
# prunetrain(model, quick_pruning=True, data='coco.yaml', train_epochs=10, imgsz=640, batch=8, device=[0], name='yolo11',
#            prune_ratio=0.5, prune_iterative_steps=1)

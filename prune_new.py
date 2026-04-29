import gc
import math
import multiprocessing
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import RANK
import torch_pruning as tp


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
                    name='prune', max_map_drop=0.10, workers=2, project='./runs/detect'):
    per_step_ratio = 1 - (1 - target_prune_ratio) ** (1 / iterative_steps)

    init_map = model.val(data=data, imgsz=imgsz, batch=batch, device=device, project=project, workers=workers).box.map
    gc.collect()
    torch.cuda.empty_cache()

    # val() fuses layers inside inference_mode; reload to get non-fused, grad-capable weights
    ckpt_path = model.ckpt_path
    model.model, _ = attempt_load_one_weight(str(ckpt_path))
    model.model = model.model.float()

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
        gc.collect()
        torch.cuda.empty_cache()

        for p in model.model.parameters():
            p.requires_grad_(True)

        _train_pruned(model, name=f'{name}_step{step}', data=data,
                      epochs=finetune_epochs, imgsz=imgsz, batch=batch, device=device,
                      project=project, workers=workers)
        model.trainer = None
        gc.collect()
        torch.cuda.empty_cache()

        current_map = model.val(data=data, imgsz=imgsz, batch=batch, device=device, project=project, workers=workers).box.map
        gc.collect()
        torch.cuda.empty_cache()
        print(f'[Step {step}] mAP: {current_map:.4f} (init: {init_map:.4f})')

        if init_map - current_map > max_map_drop:
            print(f'Early stop: mAP drop {init_map - current_map:.4f} > {max_map_drop}')
            break


# Section 5 — Entry Point

def prunetrain(model: YOLO, train_epochs, prune_epochs=0, quick_pruning=True,
               prune_ratio=0.5, prune_iterative_steps=1, data='coco.yaml',
               name='yolo11', imgsz=640, batch=8, device=None,
               sparse_training=False, max_map_drop=0.10, workers=2,
               project='./runs/detect'):
    finetune_epochs = prune_epochs if not quick_pruning else train_epochs

    if not quick_pruning:
        assert train_epochs > 0 and prune_epochs > 0
        model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch,
                    device=device, name=name, project=project, prune=False,
                    sparse_training=sparse_training, workers=workers)

    iterative_prune(
        model, data=data,
        finetune_epochs=finetune_epochs,
        target_prune_ratio=prune_ratio,
        iterative_steps=prune_iterative_steps,
        imgsz=imgsz, batch=batch, device=device,
        name=name, max_map_drop=max_map_drop, workers=workers,
        project=project,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # NOTE: legacy pretrained weight는 convert_weights.py로 먼저 변환 후 사용
    # python convert_weights.py yolov8n.pt → yolov8n_c2f.pt
    model = YOLO(r'runs\detect\yolov8n_voc\weights\best.pt')
    # model = YOLO('yolov8n_c2f.pt')

    # Normal Pruning
    prunetrain(
        model,
        quick_pruning=True,       # Quick Pruning or not
        data='VOC.yaml',          # Dataset config
        train_epochs=100,           # Epochs before pruning
        prune_epochs=50,           # Epochs after pruning
        imgsz=640,                 # Input size
        batch=32,                   # Batch size
        device=[0],                # GPU devices
        name='yolov8n_voc',             # Save name
        prune_ratio=0.5,           # Pruning Ratio (50%)
        prune_iterative_steps=1,   # Pruning Iterative Steps
        sparse_training=False,     # Experimental, Allow Sparse Training Before Pruning
        workers=2,                 # 0 to avoid multiprocessing worker spawn issues on Windows
        project='./runs/detect/pruning_resume_100epoch'
    )
    # Quick Pruning (prune_epochs no need)
    # prunetrain(model, quick_pruning=True, data='coco.yaml', train_epochs=10, imgsz=640, batch=8, device=[0], name='yolov8n',
    #            prune_ratio=0.5, prune_iterative_steps=1, workers=0, sparse_training=False)

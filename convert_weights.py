"""
legacy yolov8 checkpoint → 현재 C2f (cv0/cv1 분리) 구조로 변환.

사용법:
    python convert_weights.py yolov8n.pt          # → yolov8n_c2f.pt
    python convert_weights.py yolov8n.pt out.pt   # → out.pt
"""
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
from ultralytics.utils.downloads import attempt_download_asset

BN_KEYS = ['bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var']


def convert(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location='cpu')
    old_state = (ckpt.get('ema') or ckpt.get('model')).state_dict()

    if any('cv0' in k for k in old_state):
        print(f"{src} already has cv0 keys, skipping.")
        return

    yaml_name = Path(src).stem + '.yaml'
    model = YOLO(yaml_name)
    new_state = model.model.state_dict()

    # [1] C2f 모듈 단위 처리
    # named_modules()로 C2f를 정확히 탐색해 Bottleneck 내부 cv1과 혼동하지 않음
    handled_prefixes = set()
    for name, module in model.model.named_modules():
        if not isinstance(module, C2f):
            continue
        prefix = name + '.'
        handled_prefixes.add(prefix)

        old_cv1_w_key = f'{prefix}cv1.conv.weight'
        if old_cv1_w_key not in old_state:
            continue

        old_w = old_state[old_cv1_w_key]
        half = old_w.shape[0] // 2

        # 구버전 cv1 (2*c 채널)을 cv0 (앞 절반) / cv1 (뒤 절반)으로 분리
        new_state[f'{prefix}cv0.conv.weight'] = old_w[:half]
        new_state[f'{prefix}cv1.conv.weight'] = old_w[half:]

        for bn_key in BN_KEYS:
            old_bn_k = f'{prefix}cv1.{bn_key}'
            if old_bn_k not in old_state:
                continue
            old_bn = old_state[old_bn_k]
            new_state[f'{prefix}cv0.{bn_key}'] = old_bn[:half]
            new_state[f'{prefix}cv1.{bn_key}'] = old_bn[half:]

    # [2] 나머지 레이어 (C2f 내부 포함): shape 일치 시 그대로 복사
    for k in list(new_state.keys()):
        # C2f 직속 cv0/cv1은 이미 위에서 처리 완료
        if any(k.startswith(p) for p in handled_prefixes):
            relative = k[len(next(p for p in handled_prefixes if k.startswith(p))):]
            if relative.startswith('cv0.') or relative.startswith('cv1.'):
                continue
        if k in old_state and old_state[k].shape == new_state[k].shape:
            new_state[k] = old_state[k]

    model.model.load_state_dict(new_state, strict=False)
    model.save(dst)
    print(f"Saved: {dst}")


if __name__ == '__main__':
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace('.pt', '_c2f.pt')
    src = attempt_download_asset(src)
    convert(src, dst)

import time
import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight

PATHS = {
    'original': r'runs\detect\yolov8n_voc\weights\best.pt',
    'pruned':   r'runs\detect\yolov8n_voc_step0\weights\best.pt',
}
IMGSZ = 640
WARMUP = 10
RUNS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'


def measure(name, path):
    model, _ = attempt_load_one_weight(path)
    model = model.float().eval().to(DEVICE)

    example = torch.zeros(1, 3, IMGSZ, IMGSZ).to(DEVICE)
    macs, params = tp.utils.count_ops_and_params(model, example)

    with torch.no_grad():
        for _ in range(WARMUP):
            model(example)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(RUNS):
            model(example)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / RUNS * 1000  # ms

    print(f"[{name}]")
    print(f"  Params : {params / 1e6:.3f} M")
    print(f"  MACs   : {macs / 1e9:.3f} G")
    print(f"  Latency: {elapsed:.2f} ms/img  ({1000/elapsed:.1f} FPS)")
    print()
    return params, macs, elapsed


if __name__ == '__main__':
    results = {n: measure(n, p) for n, p in PATHS.items()}

    orig  = results['original']
    pruned = results['pruned']
    print("[Speedup]")
    print(f"  Params : {orig[0]/pruned[0]:.2f}x smaller")
    print(f"  MACs   : {orig[1]/pruned[1]:.2f}x fewer")
    print(f"  Latency: {orig[2]/pruned[2]:.2f}x faster")

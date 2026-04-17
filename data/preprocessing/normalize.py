import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def compute_dataset_stats(file_paths, input_size=(224, 224)):
    """
    Compute per-channel mean and std over a list of image files.
    Useful if you want dataset-specific normalisation instead of ImageNet stats.

    Args:
        file_paths  (list[str]):  paths to image files
        input_size  (tuple):      resize target (H, W)

    Returns:
        tuple: (mean: list[float], std: list[float])  — per channel, range [0,1]
    """
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    mean = torch.zeros(3)
    sq_mean = torch.zeros(3)
    n = 0

    print(f"[Normalize] Computing stats over {len(file_paths)} images...")
    for fp in tqdm(file_paths):
        img = Image.open(fp).convert("RGB")
        t = transform(img)          # [3, H, W]
        mean   += t.mean(dim=[1, 2])
        sq_mean += (t ** 2).mean(dim=[1, 2])
        n += 1

    mean    = (mean / n).tolist()
    sq_mean = (sq_mean / n).tolist()
    std = [((sq_mean[i] - mean[i] ** 2) ** 0.5) for i in range(3)]

    print(f"[Normalize] mean={[round(m, 4) for m in mean]}")
    print(f"[Normalize] std ={[round(s, 4) for s in std]}")
    return mean, std
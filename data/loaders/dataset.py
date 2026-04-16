import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Maps EuroSAT class folder names to binary labels
# class_1_agri = 1, class_0_non_agri = 0
LABEL_MAP = {
    "class_0_non_agri": 0,
    "class_1_agri": 1,
}


class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for binary satellite image classification.

    Supports two loading modes:
      - memory:    all images loaded into RAM at init (fast iteration,
                   requires sufficient memory)
      - generator: images read from disk on each __getitem__ call
                   (memory-efficient, slightly slower)

    Args:
        file_paths  (list[str]):   absolute paths to image files
        labels      (list[int]):   corresponding binary labels (0 or 1)
        transform   (callable):    torchvision transforms to apply
        mode        (str):         "memory" or "generator"
    """

    def __init__(self, file_paths, labels, transform=None, mode="generator"):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode

        if self.mode == "memory":
            print(f"[Dataset] Loading {len(file_paths)} images into memory...")
            self.images = [
                Image.open(fp).convert("RGB") for fp in self.file_paths
            ]
            print("[Dataset] Done.")
        else:
            self.images = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.mode == "memory":
            image = self.images[idx]
        else:
            image = Image.open(self.file_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def __repr__(self):
        return (
            f"SatelliteDataset(n={len(self)}, "
            f"mode={self.mode}, "
            f"agri={sum(self.labels)}, "
            f"non_agri={len(self.labels) - sum(self.labels)})"
        )


def get_dataloaders(train_paths, train_labels,
                    val_paths,   val_labels,
                    test_paths,  test_labels,
                    train_transform, eval_transform,
                    batch_size=32, num_workers=2, mode="generator"):
    """
    Build train, validation, and test DataLoaders.

    Args:
        *_paths     (list[str]):  file paths for each split
        *_labels    (list[int]):  labels for each split
        train_transform:          augmentation + normalisation transforms
        eval_transform:           normalisation-only transforms (no augmentation)
        batch_size  (int):        samples per batch
        num_workers (int):        DataLoader worker processes
        mode        (str):        "memory" or "generator"

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_ds = SatelliteDataset(train_paths, train_labels,
                                transform=train_transform, mode=mode)
    val_ds   = SatelliteDataset(val_paths,   val_labels,
                                transform=eval_transform,  mode=mode)
    test_ds  = SatelliteDataset(test_paths,  test_labels,
                                transform=eval_transform,  mode=mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"[DataLoader] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    return train_loader, val_loader, test_loader
import os
from sklearn.model_selection import train_test_split

from data.loaders.dataset import LABEL_MAP


def collect_file_paths(data_dir):
    """
    Walk the dataset directory and collect (file_path, label) pairs.

    Expected structure:
        data_dir/
        ├── class_0_non_agri/   →  label 0
        └── class_1_agri/       →  label 1

    Args:
        data_dir (str): path to the root dataset directory

    Returns:
        tuple: (file_paths: list[str], labels: list[int])
    """
    file_paths, labels = [], []

    for class_name, label in LABEL_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected directory not found: {class_dir}"
            )
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    print(f"[Split] Found {len(file_paths)} images | "
          f"agri={sum(labels)} | non_agri={len(labels) - sum(labels)}")
    return file_paths, labels


def split_dataset(data_dir, train_ratio=0.70, val_ratio=0.15,
                  random_state=42, debug=False, debug_samples=100):
    """
    Stratified split into train / val / test sets.

    Args:
        data_dir       (str):   root dataset directory
        train_ratio    (float): proportion for training (default 0.70)
        val_ratio      (float): proportion for validation (default 0.15)
                                test ratio = 1 - train_ratio - val_ratio
        random_state   (int):   reproducibility seed
        debug          (bool):  if True, subsample to debug_samples images
        debug_samples  (int):   number of images to keep in debug mode

    Returns:
        tuple of 6 lists:
            train_paths, train_labels,
            val_paths,   val_labels,
            test_paths,  test_labels
    """
    file_paths, labels = collect_file_paths(data_dir)

    if debug:
        file_paths = file_paths[:debug_samples]
        labels     = labels[:debug_samples]
        print(f"[Split] Debug mode — using {debug_samples} samples")

    test_ratio = round(1.0 - train_ratio - val_ratio, 4)

    # first split: train vs (val + test)
    train_paths, rem_paths, train_labels, rem_labels = train_test_split(
        file_paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state,
    )

    # second split: val vs test
    val_share = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rem_paths, rem_labels,
        test_size=(1.0 - val_share),
        stratify=rem_labels,
        random_state=random_state,
    )

    print(f"[Split] train={len(train_paths)} | "
          f"val={len(val_paths)} | test={len(test_paths)}")
    return (train_paths, train_labels,
            val_paths,   val_labels,
            test_paths,  test_labels)
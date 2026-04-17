import os
from PIL import Image
from tqdm import tqdm


def tile_image(image_path, output_dir, tile_size=224, overlap=0):
    """
    Slice a large satellite image into fixed-size tiles.

    Useful if your raw data consists of large GeoTIFF scenes rather than
    pre-cropped patches. EuroSAT images are already 64×64 so this module
    is a no-op for that dataset — kept here for future use.

    Args:
        image_path  (str):  path to source image
        output_dir  (str):  directory to save tile files
        tile_size   (int):  width and height of each tile in pixels
        overlap     (int):  pixel overlap between adjacent tiles

    Returns:
        list[str]: paths to saved tile files
    """
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    stride = tile_size - overlap

    saved = []
    row, col = 0, 0
    y = 0
    while y + tile_size <= H:
        x = 0
        while x + tile_size <= W:
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            fname = f"{os.path.splitext(os.path.basename(image_path))[0]}_r{row}_c{col}.png"
            out_path = os.path.join(output_dir, fname)
            tile.save(out_path)
            saved.append(out_path)
            x += stride
            col += 1
        y += stride
        row += 1
        col = 0

    return saved


def tile_dataset(src_dir, dst_dir, tile_size=224, overlap=0):
    """
    Apply tile_image to all images in a directory tree.

    Args:
        src_dir    (str): source directory (mirrors dataset structure)
        dst_dir    (str): destination directory
        tile_size  (int): tile size in pixels
        overlap    (int): overlap in pixels
    """
    for root, _, files in os.walk(src_dir):
        for fname in tqdm(files, desc=f"Tiling {root}"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                rel = os.path.relpath(root, src_dir)
                out_dir = os.path.join(dst_dir, rel)
                tile_image(os.path.join(root, fname), out_dir, tile_size, overlap)
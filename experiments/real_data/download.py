"""Download MNIST and Fashion-MNIST (raw IDX files) and save downsampled copies.

Usage: uv run python experiments/real_data/download.py [out_dir]   (default /root/outputs/real_data/data)

Writes <out_dir>/<dataset>_raw/*.gz (the original files) and <out_dir>/<dataset>_<s>x<s>.npz with
x_train, y_train, x_test, y_test: images cropped to the central 24 x 24 pixels (MNIST digits sit in
the central 20 x 20 box), block-averaged to s x s (s = 4: 6 x 6 blocks, s = 8: 3 x 3 blocks), scaled
to [0, 1], float64.
"""

import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np

SOURCES = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def downsample(images: np.ndarray, size: int) -> np.ndarray:
    crop = images[:, 2:26, 2:26].astype(np.float64) / 255.0
    block = 24 // size
    n = crop.shape[0]
    return crop.reshape(n, size, block, size, block).mean(axis=(2, 4))


def main(out_dir: Path) -> None:
    for name, url in SOURCES.items():
        raw = out_dir / f"{name}_raw"
        raw.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for key, fname in FILES.items():
            path = raw / fname
            if not path.exists():
                print("downloading", url + fname, flush=True)
                urllib.request.urlretrieve(url + fname, path)
            arrays[key] = read_idx(path)
        for size in (4, 8):
            out = {k: (downsample(v, size) if k.startswith("x") else v.astype(np.int64))
                   for k, v in arrays.items()}
            np.savez_compressed(out_dir / f"{name}_{size}x{size}.npz", **out)
            print(name, size, {k: v.shape for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/outputs/real_data/data"))

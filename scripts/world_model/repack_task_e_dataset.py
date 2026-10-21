"""Copy a Task E HDF5 dataset with frame-aligned RGB chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve()
    if source == destination:
        raise ValueError("Source and destination must differ.")
    if destination.exists():
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as src, h5py.File(destination, "x") as dst:
        for name, value in src.attrs.items():
            dst.attrs[name] = value
        for trajectory_name in src:
            src_group = src[trajectory_name]
            dst_group = dst.create_group(trajectory_name)
            for name, value in src_group.attrs.items():
                dst_group.attrs[name] = value
            for dataset_name in ("obs", "actions", "qvel", "ee_pos", "ee_quat"):
                src_group.copy(dataset_name, dst_group)

            rgb = src_group["images/rgb"]
            output = dst_group.create_group("images").create_dataset(
                "rgb",
                shape=rgb.shape,
                dtype=rgb.dtype,
                compression="gzip",
                compression_opts=4,
                chunks=(1, *rgb.shape[1:]),
            )
            for start in range(0, len(rgb), 64):
                stop = min(start + 64, len(rgb))
                output[start:stop] = rgb[start:stop]
            print(f"[COPY] {trajectory_name}: {len(rgb)} frames", flush=True)

    print(f"[PASS] Repacked dataset: {destination}")


if __name__ == "__main__":
    main()

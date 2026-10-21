"""Select complete real pose episodes with a fixed whole-seed split."""
import argparse
import json
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    import h5py
    from demo.throw_cem.data import POSE_FORMAT, camera_contract
    from demo.throw_cem.config import ThrowConfig
    train_seeds = {43, 44, 46, 47, 48, 49, 50, 51} | set(range(70, 85)) | set(range(100, 115))
    val_seeds = {45, 52, 53} | set(range(85, 90)) | set(range(115, 120)) | set(range(140, 145))
    selected = {}
    for path in sorted((project/"work").glob("toss*/**/pose_data.hdf5")):
        with h5py.File(path, "r") as f:
            seed = int(f.attrs.get("source_seed", -1))
            if seed not in train_seeds | val_seeds:
                continue
            if (f.attrs.get("format") != POSE_FORMAT or not f.attrs.get("complete", False)
                    or json.loads(f.attrs["camera"]) != camera_contract(ThrowConfig())):
                continue
            frames = len(f["rgb"])
            if frames and frames > selected.get(seed, (0, None))[0]:
                selected[seed] = (frames, str(path.resolve()))
    sources = {"train": [selected[s][1] for s in sorted(selected) if s in train_seeds],
               "validation": [selected[s][1] for s in sorted(selected) if s in val_seeds],
               "split_policy": "complete episode seeds; no frame-wise split"}
    destination = Path(args.output_file)
    if destination.exists():
        parser.error("Source manifests are immutable; choose a new path")
    destination.write_text(json.dumps(sources, indent=2))
    print(json.dumps({"train_episodes": len(sources["train"]), "val_episodes": len(sources["validation"]),
                      "raw_frames": sum(n for n,_ in selected.values())}), flush=True)


if __name__ == "__main__":
    main()

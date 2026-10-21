"""CLI argument definitions for Task E demo collection.
"""


def add_collect_demo_args(parser) -> None:
    parser.add_argument("--native_physics", action="store_true",
        help="Keep original Task-E actuators for aligned world-model collection.")
    parser.add_argument("--aligned", action="store_true",
        help="Record exact 24-D/RGB transitions and isolated training labels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery_mix", action="store_true", help="Six bounded training-only fault/retry scenarios; keeps failures")
    parser.add_argument("--max_record_steps", type=int, default=6000, help="Recovery-mode hard per-episode record budget")
    parser.add_argument(
        "--num_demos", type=int, default=50,
        help="Number of saved episodes; failures retained unless --only_success.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./datasets/atec_task_e",
        help="Directory to save trajectory.hdf5 and trajectory.json.",
    )
    parser.add_argument(
        "--pick_objects", type=int, nargs="+", default=[3, 2, 1],
        metavar="N",
        help="Ordered objects to pick (subset of {1,2,3}). "
        "Default: 3 2 1 (nearest to farthest from the basket).",
    )
    parser.add_argument(
        "--save_video", action="store_true", default=False,
        help="Save an MP4 per demo for visualization "
             "(requires: pip install imageio imageio-ffmpeg).",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Output directory for MP4 files. Defaults to <output_dir>/videos/.",
    )
    parser.add_argument(
        "--save_images", action="store_true", default=False,
        help="Save raw RGB frames into HDF5 under traj_N/images/rgb (T,H,W,3) "
             "for ACT RGBD training. Shares the camera with --save_video.",
    )
    parser.add_argument(
        "--image_size", type=int, default=0,
        help="Optional square RGB size stored in HDF5/video. Zero keeps the native camera resolution.",
    )
    parser.add_argument(
        "--only_success", action="store_true", default=False,
        help="Discard demos where not all picked objects ended up in the basket.",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=0,
        help="Maximum collection attempts. Zero uses 10 times --num_demos.",
    )

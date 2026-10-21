"""Select the Task E policy backend without changing the evaluator import."""

import os


_backend = os.environ.get("ATEC_TASK_E_BACKEND", "act").strip().lower().replace("-", "_")
if _backend == "act":
    from .solution_act import AlgSolution
elif _backend in {"world", "world_model", "wm"}:
    from .solution_world_model import AlgSolution
elif _backend == "jepa":
    from .solution_jepa import AlgSolution
else:
    raise ValueError(
        f"Unsupported ATEC_TASK_E_BACKEND={_backend!r}; expected 'act', 'world_model', or 'jepa'."
    )

__all__ = ["AlgSolution"]

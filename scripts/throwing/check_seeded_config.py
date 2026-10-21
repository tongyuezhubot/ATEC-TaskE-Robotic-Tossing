"""One config-only check: no environment, renderer, trajectory or training."""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg
    def positions(cfg):
        return tuple(tuple(getattr(cfg.scene,f"object_{i}").init_state.pos) for i in (1,2,3))
    first = positions(TaskEEnvPiperCfg(seed=290))
    repeated = positions(TaskEEnvPiperCfg(seed=290))
    other = positions(TaskEEnvPiperCfg(seed=291))
    assert first == repeated and first != other
    print("SEEDED_CONFIG_PASS: same-seed layout identical; different seed changes layout",flush=True)
finally:
    app.close()

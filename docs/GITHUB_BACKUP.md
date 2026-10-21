# Project backup — 2026-10-21

This repository is a source-code snapshot of the server project
`/root/Desktop/ATEC_TaskE_Standalone`. It includes the original task environment,
ACT and experimental learned-model code, the current throwing controller,
tests, documentation, robot assets, and explicitly selected small runtime JSON
configurations and dynamics weights.

## Current working approach

The throwing controller combines a scripted grasping state machine, calibrated
joint-response dynamics, a physical release/flight prior, a per-object learned
release residual, and CEM motion planning. The residual model uses regularized
linear regression, not a general-purpose JEPA or video world model.

The three-object development demonstration uses privileged simulator object
poses. It is not a completed RGB-only policy or a formal success-rate evaluation.
Existing ACT and JEPA experiments are retained as historical code, not claims
about the active throwing controller.

## Included runtime files

`work/throw_profiles_chain_2026-10-15.json` selects the three motion profiles and
release models. The small JSON files retain their original relative paths so
that existing server scripts can locate them. Nested runtime JSON files are
explicitly tracked despite the default work-directory exclusion.

The original `README.md`, `ORIGIN_MANIFEST.json`, and `SHA256SUMS` describe the
upstream/original delivery. Their checksums are not checksums of this snapshot.

## Not included

- SSH keys, passwords, access tokens, and local authentication settings.
- Raw datasets, replay buffers, feature caches, generated logs, and training runs.
- Large ACT, JEPA, DINO, or RGB pose checkpoints (`*.pt`, `*.pth`).
- The server's Git history or Git LFS objects.

This is therefore a code/configuration backup, not a complete server image.
Scripts referring to omitted datasets or vision weights require those files to
be restored separately. Current Oracle chain execution does not use the RGB
pose checkpoints, but RGB execution requires them and sugar RGB remains pending.

Isaac Sim / Isaac Lab and other third-party dependencies must be installed
separately. Some launch scripts retain the existing server installation paths.
No simulation or training job was started as part of this backup.

# Repository instructions for coding agents

This repository is the shared codebase for a three-person team entering the AIC 2026 UAV
semantic-segmentation challenge. All three teammates are equal collaborators; do not assume
fixed roles or a single permanent operator.

## Start every task by reading

1. `README.md`
2. `docs/HANDOFF.md`
3. `docs/EXPERIMENTS.md`
4. `git status` and the latest commits

Before changing code, summarize the current state, identify the one experiment or defect being
addressed, and preserve unrelated work.

## Competition constraints

- The final prediction must come from one model. Do not implement model ensembles or weight/prediction averaging.
- Publicly available academic pretrained weights may be fine-tuned.
- Do not use commercial closed-model APIs for competition inference.
- The organizer must be able to reproduce the result from the submitted code and official data.
- Label IDs are currently `0-8`; ID `0` is ignored by the local metric. Re-check this against every official data release.
- Do not assume that test-set class or domain distributions match the training set.

## Repository and data safety

- Never commit official datasets, test images, predictions, checkpoints, SSH credentials, API tokens, or passwords.
- Keep generated data under ignored directories such as `data/`, `outputs/`, `checkpoints/`, and `submissions/`.
- Do not weaken `.gitignore` to publish artifacts.
- Treat existing user changes as intentional. Do not reset, delete, or overwrite them.
- Record dependency versions, random seeds, data splits, configuration, metrics, and the Git commit for every real experiment.

## Development expectations

- Keep the baseline reproducible and single-model.
- Run `python -m unittest discover -s tests -v` after relevant changes.
- Prefer one controlled change per experiment so that an mIoU change has a clear cause.
- Do not claim an improvement from the 11-image smoke dataset; it verifies the pipeline only.
- Formal validation must avoid leakage between neighboring patches from the same source image, scene, or flight.
- Update `docs/HANDOFF.md` before handing work to another teammate or Codex.
- Append real experiments to `docs/EXPERIMENTS.md`; never rewrite past results to make them look better.

## AutoDL conventions

All large or generated files belong on the data disk:

```text
/root/autodl-tmp/aicomp/
├── repo/
├── datasets/
├── models/
├── checkpoints/
├── outputs/
└── logs/
```

Long-running commands must use `screen` (or an equivalent persistent session) and write logs to
the data disk. An SSH disconnect must not terminate training. Shutting down a pay-as-you-go
instance releases its GPU; never recommend shutdown without stating that the card may be
unavailable at the next start.

When switching instances, verify the copied data disk before releasing the old instance. A new
instance has new SSH connection details even when it was cloned.


# Project conventions

- Use Python 3.11 in `.venv`; install the exact pins with `make setup` from WSL2.
- Run commands from the repository root. Keep datasets, generated metrics, model files, and runtime databases out of Git.
- Every reported metric must be produced by a run and saved under `results/`; never use illustrative numbers as results.
- Keep transaction ordering and causal feature calculations intact. Add focused pytest coverage for each phase.
- Complete only the currently authorized phase and document environment checks that could not be run as `UNVERIFIED` in `docs/phase_N_notes.md`.


# Benchmark Template

This directory provides scaffold files for adding a new benchmark to prompt-opt.

## Files

- `env_template.py` — Environment adapter template
- `loader_template.py` — Data loader template
- `config_template.yaml` — Config file template

## Usage

1. Copy this directory: `cp -r promptopt/envs/_template promptopt/envs/your_benchmark`
2. Rename files: remove `_template` suffix
3. Implement the `TODO` sections
4. Register in `promptopt/envs/__init__.py`
5. Create config at `configs/your_benchmark/default.yaml`

See the [documentation](../../docs/guide/new-benchmark.md) for the full guide.

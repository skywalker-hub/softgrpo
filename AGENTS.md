# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

SofT-GRPO is a GPU-only ML research project for LLM reinforcement learning. It has two main packages installed in editable mode:

- **SGLang** (modified): `Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg/` — LLM serving engine with soft-thinking support
- **verl** (modified): `verl-0.4.x/` — RL training framework (Volcano Engine RL)

### Python environment

- Python 3.11 virtual environment at `/workspace/.venv`
- Always activate with `source /workspace/.venv/bin/activate` before running any commands
- PyTorch is installed as CPU-only (`torch==2.6.0+cpu`) since the Cloud VM has no GPU
- GPU-only packages (`flash-attn`, `vllm`, `sgl-kernel` CUDA builds) cannot be installed without GPU hardware

### Linting

- **verl**: `cd /workspace/verl-0.4.x && ruff check .` (config in `pyproject.toml`)
- **sglang**: `cd /workspace/Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg && ruff check python/sglang/`
- Both projects have pre-commit configs (`.pre-commit-config.yaml`) — verl uses ruff, sglang uses ruff + black + isort
- Pre-existing lint warnings exist in both codebases (67 in verl, 433 in sglang) — these are not regressions

### Testing

- **CPU tests (verl)**: `cd /workspace/verl-0.4.x && python -m pytest tests/sanity/ tests/test_protocol.py tests/trainer/ppo/test_metric_utils.py tests/utils/cpu_tests/ -v`
- **Ray CPU tests (verl)**: `cd /workspace/verl-0.4.x && python -m pytest tests/ray_cpu/ -v` (note: `test_check_worker_alive` has a pre-existing path bug)
- GPU tests (`tests/ray_gpu/`, `tests/gpu_utility/`, `tests/kernels/`, `tests/models/`) require NVIDIA GPUs and cannot run in this environment
- sglang kernel tests (`sgl-kernel/tests/`) all require GPU

### Running the application

- The SGLang server (`python -m sglang.launch_server`) requires NVIDIA GPU with CUDA 12.4 and the `vllm` package
- Training scripts (`SofT-GRPO-deepscaler-8k*.sh`) require 8x NVIDIA GPUs
- Model weights must be downloaded from HuggingFace before inference/training
- The SGLang client-side API (`sglang.api`, `sglang.lang.*`) works without GPU and can be used for function definitions and remote calls

### Key gotchas

- `pytest --timeout` flag is not supported — the timeout plugin is not installed
- The `tensordict` version must be `<=0.6.2` for verl compatibility
- `antlr4-python3-runtime` version differs between sglang (4.13.2) and verl/hydra (4.9.3) — verl's version takes precedence during editable install
- Datasets are in `Soft-Thinking+noise+loss-main/datasets/` (parquet + json formats)

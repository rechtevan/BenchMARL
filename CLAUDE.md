# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BenchMARL is a Multi-Agent Reinforcement Learning (MARL) training library built on TorchRL. It provides standardized
interfaces for benchmarking different MARL algorithms and environments with reproducible configurations.

**Core architecture components:**

- **Experiment**: A single training run with fixed algorithm, task, and model
- **Benchmark**: A collection of experiments comparing different algorithms/tasks/models
- **Algorithm**: Training strategy (loss functions, replay buffers) - implemented in `benchmarl/algorithms/`
- **Task**: Environment scenarios to solve - implemented in `benchmarl/environments/`
- **Model**: Neural network architectures (actors/critics) - implemented in `benchmarl/models/`

**Key technologies:**

- TorchRL as the RL backend
- Hydra for configuration management
- marl-eval compatible output format for plotting

## Fork & Contribution Context

This is a fork by rechtevan (Evan Recht) focused on code quality improvements. Changes developed in this fork will be
submitted as pull requests to the upstream repository.

**Repository information:**

- Fork: https://github.com/rechtevan/BenchMARL
- Upstream: https://github.com/facebookresearch/BenchMARL
- Maintainer: Meta Platforms, Inc.
- License: MIT License

**Contribution workflow:**

1. Develop improvements in fork with comprehensive testing
2. Submit PRs to upstream facebookresearch/BenchMARL
3. Maintain attribution via git history (no per-file copyright headers needed)
4. Follow upstream contribution guidelines and code review process

## Code Quality Infrastructure

BenchMARL follows a structured approach to code quality with automated tooling and clear coverage targets.

**Directory conventions:**

- `.local/` - AI-generated artifacts, temporary files, coverage reports (gitignored)
- `.local/coverage/` - HTML and JSON coverage reports
- `.local/analysis/` - Code quality analysis outputs
- Never commit `.local/` contents to version control

**Testing strategy:**

- Target: 90% overall coverage / 80% minimum for all modules
- **Core code (90% coverage required):**
  - `benchmarl/algorithms/` - Training algorithms
  - `benchmarl/models/` - Neural network architectures
  - `benchmarl/experiment/` - Experiment orchestration
  - `benchmarl/*.py` - Core utilities
- **Environment wrappers (80% minimum):**
  - `benchmarl/environments/` - Lower priority due to external dependencies
- **Excluded from coverage targets:**
  - Visualization and plotting code
  - `examples/` directory
  - `docs/` directory
  - Experimental/unstable features

**Coverage tooling:**

- pytest with coverage.py integration
- HTML reports for local review
- JSON reports for CI/CD integration
- Codecov integration for PR coverage tracking

## Code Quality Tools

BenchMARL uses a comprehensive suite of code quality tools to maintain high standards.

**Linting and formatting:**

- **Current:** flake8 for style enforcement
- **Recommended modernization:** Migrate to Ruff (faster, more comprehensive)
  - Ruff combines flake8, isort, pyupgrade, and more
  - 10-100x faster than existing tools
  - License: MIT (compatible)

**Type checking:**

- Add MyPy for static type analysis
- Gradual typing approach: prioritize public APIs first
- Type hints for all new public functions and classes
- Configuration in `pyproject.toml` or `mypy.ini`

**Pre-commit hooks:**

- Already configured in repository
- Runs automatically before commits
- Enforces consistent code style
- Prevents committing common issues

**Markdown quality:**

- mdformat - Markdown formatting consistency
- pymarkdownlnt - Markdown linting
- Ensures documentation quality

**License compliance:**

- All runtime dependencies: MIT, Apache 2.0, or BSD licenses only
- Development tools can use other open-source licenses
- Verify licenses before adding new dependencies

## CI/CD & Security

BenchMARL maintains robust continuous integration and security practices.

**Current CI/CD infrastructure:**

- GitHub Actions workflows
- Unit test execution on multiple Python versions
- Codecov integration for coverage tracking
- Docker-based testing for reproducibility

**Recommended additions:**

- **CodeQL security scanning:**
  - Automated vulnerability detection
  - Scans for common security patterns
  - Integrated into GitHub Security tab
- **Parallel job execution:**
  - Fast feedback on PRs (< 5 minutes ideal)
  - Separate jobs for: linting, typing, tests, security
- **Matrix testing:**
  - Multiple Python versions (3.8, 3.9, 3.10, 3.11)
  - Different dependency versions if applicable

**Security best practices:**

- No secrets in code or configuration
- Dependency vulnerability scanning
- Regular security updates for dependencies
- CodeQL alerts reviewed promptly

## Licensing

BenchMARL is licensed under the MIT License with clear requirements for contributions and dependencies.

**Upstream license:**

- MIT License
- Copyright holder: Meta Platforms, Inc.
- Permissive license allowing commercial and private use

**Fork licensing:**

- Maintains MIT License from upstream
- Attribution preserved via git commit history
- No per-file copyright headers required
- Contributors retain copyright while granting MIT license rights

**Dependency requirements:**

- **Runtime dependencies:** MIT, Apache 2.0, or BSD only
- **Development dependencies:** Any OSI-approved open-source license
- **Rationale:** Ensure downstream users have maximum flexibility

**License verification:**

- Check licenses before adding dependencies
- Use `pip-licenses` or similar tools
- Document any exceptions or special cases

## Development Workflow

Follow these practices to maintain code quality and consistency.

**Using .local/ directory:**

- All temporary and generated files go in `.local/`
- Coverage reports: `.local/coverage/html/` and `.local/coverage/coverage.json`
- Analysis outputs: `.local/analysis/`
- Never commit `.local/` to version control
- Add to `.gitignore` if not already present

**Pre-commit workflow:**

1. Make code changes
2. Pre-commit hooks run automatically on `git commit`
3. If hooks fail: fix issues and recommit
4. If hooks pass: commit proceeds

**Pull request requirements:**

- All tests passing (CI green checkmark)
- Coverage maintained or improved (no regression)
- Pre-commit hooks passing (linting clean)
- Type hints added for new public APIs
- Documentation updated if behavior changes
- No security issues from CodeQL

**Code review focus areas:**

- Correctness of algorithm implementations
- Proper handling of multi-agent dimensions
- TorchRL API usage patterns
- Configuration schema compatibility
- Test coverage for new features

## Python-Specific Guidelines for BenchMARL

BenchMARL has specific patterns and practices due to its TorchRL/PyTorch foundation.

**TorchRL/PyTorch patterns:**

- Use `TensorDict` for batched multi-agent data
- Models inherit from `TensorDictModule` or `TensorDictSequential`
- Always handle batch dimensions correctly
- Use proper device placement (`train_device`, `sampling_device`)
- Leverage TorchRL's specs system for input/output validation

**Hydra configuration management:**

- All configs are YAML files loaded into dataclasses
- Use `@dataclass` with type annotations for config classes
- Implement `get_from_yaml()` methods for component loading
- Never modify default configs in `benchmarl/conf/` for reproducibility
- Override via command line: `python run.py algorithm.lr=0.01`

**Type hints for public APIs:**

- All public functions/methods must have type hints
- Include return type annotations
- Use `torch.Tensor` or `tensordict.TensorDict` for tensor types
- Import types from `typing` (List, Dict, Optional, etc.)
- Example:
  ```python
  def get_loss_and_updater(
      self, group: str
  ) -> Tuple[LossModule, Dict[str, TargetNetUpdater]]: ...
  ```

**Dataclass-based configs for validation:**

- Config classes use `@dataclass` decorator
- Leverage type annotations for automatic validation
- Use default values and field validators
- Example:
  ```python
  @dataclass
  class AlgorithmConfig:
      lr: float = 0.001
      gamma: float = 0.99
      batch_size: int = 256
  ```

**Multi-agent dimension handling:**

- Check `input_has_agent_dim` flag
- Handle both shared and per-agent parameters
- Use proper reshaping for multi-agent batches
- Test with different agent group configurations

## Development Commands

### Installation

```bash
# Install TorchRL first
pip install torchrl

# Install BenchMARL (editable mode for development)
git clone https://github.com/facebookresearch/BenchMARL.git
pip install -e BenchMARL

# Optional environment dependencies
pip install vmas  # VMAS simulator
pip install "pettingzoo[all]"  # PettingZoo
pip install dm-meltingpot  # MeltingPot
pip install git+https://github.com/Farama-Foundation/MAgent2  # MAgent2
```

### Running Experiments

```bash
# Basic single experiment
python benchmarl/run.py algorithm=mappo task=vmas/balance

# Multi-run benchmark (sequential by default)
python benchmarl/run.py -m algorithm=mappo,qmix,masac task=vmas/balance,vmas/sampling seed=0,1

# Override experiment config
python benchmarl/run.py algorithm=mappo task=vmas/balance experiment.lr=0.03 experiment.evaluation=true

# Resume from checkpoint
python benchmarl/resume.py path/to/checkpoint_100.pt

# Evaluate checkpoint
python benchmarl/evaluate.py path/to/checkpoint_100.pt
```

### Testing

```bash
# Run all tests with coverage
pytest test/ --doctest-modules --junitxml=junit/test-results.xml --cov=. --cov-report=xml --cov-report=html

# Run tests with enhanced coverage reporting (to .local/)
pytest test/ --cov=benchmarl --cov-report=html:.local/coverage/html --cov-report=json:.local/coverage/coverage.json --cov-report=term

# Run specific test file
pytest test/test_vmas.py

# Run specific test
pytest test/test_models.py::test_mlp_model

# Run tests for core modules only (high coverage targets)
pytest test/ --cov=benchmarl/algorithms --cov=benchmarl/models --cov=benchmarl/experiment --cov-report=term-missing
```

**Coverage targets:**

- **Overall target: 90% coverage**
- **Core modules (90% minimum required):**
  - `benchmarl/algorithms/` - All algorithm implementations
  - `benchmarl/models/` - Neural network architectures
  - `benchmarl/experiment/` - Experiment orchestration and training loops
  - `benchmarl/*.py` - Core utilities
- **Environment wrappers (80% minimum):**
  - `benchmarl/environments/` - Lower priority due to external dependencies
- **Excluded from coverage requirements:**
  - Visualization/plotting code
  - `examples/` directory
  - `docs/` directory
  - Experimental or unstable features

**Coverage report locations:**

- HTML: `.local/coverage/html/index.html` (open in browser)
- JSON: `.local/coverage/coverage.json` (for CI integration)
- Terminal: Shows coverage summary and missing lines

### Building Documentation

```bash
cd docs
make html  # Generate HTML documentation
```

## Configuration System

BenchMARL uses Hydra for hierarchical configuration management:

**Configuration hierarchy:**

```
benchmarl/conf/
├── config.yaml          # Root config with defaults
├── experiment/          # Training hyperparameters (lr, batch size, etc.)
├── algorithm/           # Algorithm-specific configs (mappo, qmix, etc.)
├── task/                # Environment tasks (vmas/, pettingzoo/, etc.)
└── model/               # Neural network architectures
    └── layers/          # Individual layer configs (mlp, gnn, cnn, etc.)
```

**Configuration loading:**

- YAML configs are loaded into Python dataclasses for type validation
- Each component has a `get_from_yaml()` method to load its config
- Hydra allows override via command line: `algorithm.lr=0.01`
- Configs are versioned with the library for reproducibility

**Key files:**

- `benchmarl/conf/experiment/base_experiment.yaml` - experiment hyperparameters
- `benchmarl/hydra_config.py` - Hydra loading logic
- Component configs loaded via `*Config` dataclasses in each module

## Core Architecture Details

### Experiment Flow

1. **Initialization** (`benchmarl/experiment/experiment.py`):

   - Load task → create TorchRL environment with transforms
   - Instantiate algorithm → build models (actor/critic)
   - Setup collectors, replay buffers, optimizers

2. **Training Loop**:

   - Collect rollouts from environment
   - Process batch through algorithm's loss modules
   - Update models via backpropagation
   - Periodic evaluation and checkpointing

3. **Key methods:**

   - `Experiment.run()` - main training loop
   - `Experiment.reload_from_file()` - resume from checkpoint

### Algorithm Implementation

Algorithms inherit from `benchmarl/algorithms/common.py:Algorithm` base class.

**Required methods to implement:**

- `get_loss_and_updater()` - returns loss module and target network updaters
- `get_replay_buffer()` - configures replay buffer (for off-policy)
- `get_policy_for_loss()` - policy module for training
- `get_policy_for_collection()` - policy for environment interaction

**Parameter sharing:**

- Controlled by `experiment.share_policy_params` config
- Each algorithm handles both shared and independent parameters
- Uses TorchRL's group-based API for multi-agent scenarios

### Task/Environment System

Tasks are defined as enumerations per environment (e.g., `VmasTask.BALANCE`).

**Task registration** (`benchmarl/environments/__init__.py`):

- `task_config_registry` maps task names to configurations
- Each environment has a `TaskClass` defining `get_env_fun()`
- Tasks return TorchRL `EnvBase` with proper transforms applied

**Multi-agent grouping:**

- Uses TorchRL MARL API for agent groups
- Competitive envs have separate groups (e.g., red vs blue team)
- Groups share policies/buffers within group, not across

### Model System

Models process agent observations and output actions/values.

**Key dimensions:**

- `input_has_agent_dim`: Whether input has multi-agent dimension
- `centralised`: Whether model has full observability
- `share_params`: Single parameter set vs per-agent parameters
- `output_has_agent_dim`: Computed from above - output shape depends on these flags

**Sequence models:**

- `SequenceModelConfig` chains multiple model layers
- Configure via Hydra: `model=sequence "model/layers@model.layers.l1=mlp"`
- Intermediate sizes control layer output dimensions

## Important Patterns

### Adding a New Algorithm

1. Create file in `benchmarl/algorithms/your_algo.py`
2. Define `YourAlgoConfig` dataclass with hyperparameters
3. Implement `YourAlgo(Algorithm)` with required abstract methods
4. Register in `benchmarl/algorithms/__init__.py`
5. Add YAML config in `benchmarl/conf/algorithm/your_algo.yaml`
6. See `examples/extending/algorithm/` for full example

### Adding a New Task

1. Create module in `benchmarl/environments/your_env/task_name.py`
2. Implement `TaskClass` with `get_env_fun()` method
3. Define task enum in `__init__.py`
4. Register task in task registry
5. Add YAML config in `benchmarl/conf/task/your_env/task_name.yaml`
6. See `examples/extending/task/` for full example

### Adding a New Model

1. Create file in `benchmarl/models/your_model.py`
2. Define `YourModelConfig` dataclass
3. Implement `YourModel(Model)` with `_forward()` method
4. Handle agent dimensions correctly (check `input_has_agent_dim`, `share_params`, `centralised`)
5. Register in `benchmarl/models/__init__.py`
6. Add YAML config in `benchmarl/conf/model/layers/your_model.yaml`
7. See `examples/extending/model/` for full example

### Working with TorchRL

BenchMARL heavily uses TorchRL primitives:

- `TensorDict`: Dictionary of tensors with batch dimensions
- `TensorDictModule`: Neural network that operates on TensorDicts
- `EnvBase`: Base environment class with standardized API
- `Transform`: Environment wrappers for preprocessing
- Specs define input/output shapes and domains

**Common TorchRL objects:**

- `SyncDataCollector`: Collects rollouts from environments
- `ReplayBuffer`: Stores transitions for off-policy learning
- `LossModule`: Computes losses from batch data
- `TargetNetUpdater`: Updates target networks (soft/hard)

### Checkpointing and Resuming

- Checkpoints saved every `experiment.checkpoint_interval` frames
- Checkpoint contains: models, optimizers, collectors, RNG states
- Resume preserves exact training state including exploration schedule
- Hydra metadata in `.hydra/` folder required for full reload

### Logging and Evaluation

- Loggers: wandb, csv, mlflow, tensorboard (via TorchRL)
- Output JSON compatible with marl-eval for plotting
- Evaluation runs separate episodes without exploration
- Use `experiment.evaluation=true` and set `experiment.evaluation_interval`

## Testing Guidelines

Tests use pytest fixtures from `test/conftest.py`:

- `experiment_config`: Pre-configured with small values for fast testing
- Model sequence configs: Various model architecture combinations

**Test structure:**

- `test/test_*.py` files per environment/component
- Integration tests run actual training for few iterations
- Model tests verify forward pass shapes and gradient flow

**Coverage requirements:**

- **Overall target: 90% coverage**
- **Core modules (90% minimum required):**
  - `benchmarl/algorithms/` - Algorithm implementations and loss functions
  - `benchmarl/models/` - Neural network architectures and forward passes
  - `benchmarl/experiment/` - Training loops, checkpointing, evaluation
  - `benchmarl/*.py` - Core utilities
- **Environment wrappers (80% minimum):**
  - `benchmarl/environments/` - Lower priority due to external dependencies
- **Excluded from coverage targets:**
  - Plotting and visualization utilities
  - `examples/` directory (example scripts)
  - `docs/` directory (documentation)
  - Experimental features marked as unstable

**Coverage reporting:**

- Generate reports to `.local/coverage/` directory
- HTML reports: `.local/coverage/html/index.html`
- JSON reports: `.local/coverage/coverage.json`
- Use pytest coverage flags: `--cov=benchmarl --cov-report=html:.local/coverage/html`
- Review coverage locally before submitting PRs
- CI tracks coverage and reports regressions

**Writing effective tests:**

- Test both success and failure cases
- Verify tensor shapes and dimensions
- Test with different agent group configurations
- Test parameter sharing on/off
- Use small problem sizes for fast execution
- Mock expensive operations when appropriate

## Code Quality Checklist

Use this checklist when preparing pull requests to ensure code quality standards are met.

**Pre-submission checklist:**

- [ ] All tests pass locally (`pytest test/`)
- [ ] Coverage maintained or improved (check `.local/coverage/html/index.html`)
- [ ] Pre-commit hooks pass (run `pre-commit run --all-files`)
- [ ] Type hints added for new public APIs
- [ ] Documentation updated if behavior changes
- [ ] No new security issues (if CodeQL is enabled)
- [ ] All new dependencies have compatible licenses (MIT/Apache 2.0/BSD)
- [ ] Code follows BenchMARL patterns (TorchRL, Hydra, multi-agent dimensions)

**Algorithm-specific checklist:**

- [ ] Implements all required abstract methods from `Algorithm` base class
- [ ] Handles both shared and independent parameter modes
- [ ] Works with multi-agent groups correctly
- [ ] Configuration dataclass defined with proper types
- [ ] YAML config file added in `benchmarl/conf/algorithm/`
- [ ] Tests cover both on-policy and off-policy scenarios (if applicable)
- [ ] Replay buffer configuration correct (for off-policy algorithms)

**Model-specific checklist:**

- [ ] Handles `input_has_agent_dim`, `share_params`, `centralised` flags correctly
- [ ] Output shapes verified with different configurations
- [ ] Works in both shared and per-agent parameter modes
- [ ] Configuration dataclass and YAML config added
- [ ] Tests verify gradient flow and tensor shapes
- [ ] Compatible with sequence model composition

**Task/Environment-specific checklist:**

- [ ] Returns proper TorchRL `EnvBase` instance
- [ ] Specs defined correctly (observation, action, reward)
- [ ] Multi-agent grouping handled appropriately
- [ ] Configuration dataclass and YAML config added
- [ ] Tests verify environment reset and step functions
- [ ] Compatible with existing algorithms

**Documentation checklist:**

- [ ] Docstrings for public functions and classes
- [ ] Type hints in function signatures
- [ ] README updated if adding new features
- [ ] Examples added for new components (in `examples/extending/`)
- [ ] Configuration options documented

**Performance checklist (for significant changes):**

- [ ] No unnecessary tensor copies
- [ ] Proper device placement
- [ ] Batch operations where possible
- [ ] Memory usage reasonable for typical scenarios

## Project Conventions

**Code organization:**

- Each component (algorithm/task/model) has `common.py` with base classes
- Specific implementations in separate files
- `__init__.py` handles registration and exports

**Configuration versioning:**

- Default configs in `benchmarl/conf/` are versioned with library releases
- Task configs should NOT be modified for reproducibility
- Experiment configs can be freely customized

**Group-based design:**

- All algorithms/models work with agent groups (TorchRL MARL API)
- Even single-group scenarios use group abstraction
- Parameter sharing is within-group, not across groups

**Device handling:**

- `experiment.sampling_device`: Where environments run
- `experiment.train_device`: Where model training happens
- `experiment.buffer_device`: Where replay buffer is stored
- Can be same or different (e.g., "cpu", "cuda:0")

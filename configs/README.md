# EDG Configuration System

This directory contains YAML configuration files for EDG (Ensembles from Density Generator) experiments. The configuration system supports both single experiments and **multi-GPU parallel batch execution**, replacing hardcoded parameters in `run_*.py` scripts with flexible, version-controlled YAML files.

## Quick Start

### Basic Usage
```bash
# Run a single experiment
edg --config configs/simple_experiment.yaml

# Run multiple experiments in parallel across GPUs
edg --config configs/batch_experiment.yaml --max-parallel 4

# Override parameters from command line
edg --config configs/ptp1b.yaml --num-steps 300 --resolution 1.5

# Validate configuration without running
edg --config configs/mac1.yaml --validate-only
```

### Creating Your First Experiment
1. **Single experiment**: `cp configs/templates/simple_experiment.yaml configs/my_experiment.yaml`
2. **Batch experiments**: `cp configs/templates/batch_experiment.yaml configs/my_batch.yaml`
3. Edit the paths and parameters in the YAML file
4. Run: `edg --config configs/my_experiment.yaml`

## Directory Structure

```
configs/
├── README.md                    # This file
├── templates/                   # Template configurations
│   ├── basic_experiment.yaml    # Full-featured single experiment template
│   ├── simple_experiment.yaml   # Simple single experiment template
│   └── batch_experiment.yaml    # Multi-GPU batch experiment template
├── ptp1b.yaml                  # PTP1B experiment (replaces run_ptp1b.py)
├── mac1.yaml                   # MAC1 experiment (replaces run_mac1.py)
├── synthetic_density_test.yaml # Large-scale batch experiment example
└── your_experiment.yaml        # Your custom experiments
```

## Configuration Types

### Single Experiment Configuration

For running a single EDG experiment:

#### Required Sections

##### Experiment Identity
```yaml
name: "my_experiment"          # Experiment name for outputs
```

##### Structure Configuration
```yaml
structure:
  structure_path: "/path/to/structure.cif"    # Input structure file
  clean_structure: true                       # Apply structure cleaning
  keep_type: "protein"                        # "protein", "all"
  remove_all_ligands: true                    # Remove all ligands
  remove_alternative_conformations: true      # Remove alt conformations
  complete_residues: true                     # Complete missing atoms
```

##### Density Configuration
```yaml
density:
  map_path: "/path/to/density.ccp4"          # Density map file
  resolution: 2.0                             # Resolution (required for CCP4/MRC)
  em_mode: false                              # Use EM scattering factors
```

##### Output Configuration
```yaml
output_dir: "results/my_experiment"          # Where results are saved
input_data_dir: "input/my_experiment"        # Temporary input files
```

### Batch Experiment Configuration

For running multiple experiments in parallel across GPUs:

#### Required Sections

##### Batch Identity
```yaml
name: "batch_experiment"                     # Batch name for outputs
max_parallel: 4                             # Number of parallel experiments (GPUs)
continue_on_error: true                      # Continue batch on experiment failures
output_base_dir: "results/batch"             # Base output directory
input_base_dir: "input/batch"                # Base input directory
```

##### Shared Configuration
```yaml
shared_config:
  # Base configuration applied to all experiments
  structure:
    clean_structure: true
    keep_type: "protein"
    remove_all_ligands: true
  density:
    resolution: 2.0
  model:
    version: "boltz2"
  # ... other shared parameters
```

##### Experiments List
```yaml
experiments:
  - name: "exp1"
    structure:
      structure_path: "/path/to/structure1.cif"
    density:
      map_path: "/path/to/density1.ccp4"
    output_dir: "results/batch/exp1"
    input_data_dir: "input/batch/exp1"
    # Override any shared parameters
    adaptive_solver:
      learning_rate: 0.01
  - name: "exp2"
    structure:
      structure_path: "/path/to/structure2.cif"
    density:
      map_path: "/path/to/density2.ccp4"
    output_dir: "results/batch/exp2"
    input_data_dir: "input/batch/exp2"
    # Different overrides
    density_guidance:
      base_weight: 0.8
```

### Optional Sections (with defaults)

#### Model Configuration
```yaml
model:
  version: "boltz2"                           # "boltz1" or "boltz2"
  checkpoint_path: null                       # Auto-detect if null
  device: null                                # Auto-detect if null
  pre_loaded_model: null                      # Pre-loaded model instance
```

#### Diffusion Parameters
```yaml
diffusion:
  num_steps: 200                              # Number of diffusion steps
  step_scale: null                            # Auto-set based on model
  noise_scale: 1.0
  gamma_0: 1.0
  gamma_min: 0.01
```

#### Steering Configuration
```yaml
steering:
  enabled: true                               # Enable steering
  guidance_update: true                       # Update guidance during optimization
  num_particles: 3                            # Number of particles for steering
  fk_resampling_interval: 1                   # Resampling frequency
  fk_lambda: 0.5                              # Lambda parameter for steering
  num_gd_steps: 10                            # Number of gradient descent steps
```

#### Adaptive Solver Configuration
```yaml
adaptive_solver:
  type: "adam"                                # "adam", "simple", or "none"
  learning_rate: 0.01                         # Base learning rate
  max_iterations: 20                          # Maximum gradient steps per solve
  convergence_threshold: 1e-4                 # Early stopping threshold
  gradient_clip_norm: 1.0                     # Gradient clipping limit
  per_potential_scaling: true                 # Normalize gradients per potential
  line_search: true                           # Enable backtracking line search
  adaptive_line_search: true                  # Enable adaptive line search
  # Adam-specific parameters
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
```

#### Density Guidance Configuration
```yaml
density_guidance:
  base_weight: 0.4                            # Base guidance weight (supports scheduling)
  guidance_interval: 1                        # Frequency of guidance application
  resampling_weight: 0.1                      # Weight for particle resampling
  scale_guidance_to_denoising: true           # Scale guidance to denoising ratio
  max_guidance_denoising_ratio: 0.5           # Maximum guidance/denoising ratio
  resolution: 2.0                             # Resolution for guidance (supports scheduling)
```

#### Substructure Configuration
```yaml
substructure:
  enabled: true                               # Enable substructure conditioning
  selection: "chain A and resi 120-140"       # Residue selection string
  guidance_weight: 0.05                       # Substructure guidance weight
  resampling_weight: 0.0                      # Substructure resampling weight
  buffer: 0.5                                 # Buffer distance for substructure
```

#### Potentials Configuration
```yaml
potentials:
  use_default_potentials: true                # Use default potential functions
```

#### Optimization Settings
```yaml
optimization:
  ensemble_size: 1                            # Number of ensemble members
  partial_diffusion: false                    # Use partial diffusion
  save_interval: 10                           # Save every N steps
  save_maps: true                             # Save density maps
  save_scores: true                           # Save optimization scores
```

## Parameter Scheduling

The configuration system supports complex parameter schedules that change during optimization:

### Simple Constant Values
```yaml
density_guidance:
  base_weight: 0.4                            # Constant value
```

### Piecewise Linear Schedule
```yaml
density_guidance:
  base_weight:
    breakpoints: [0.125, 0.375]               # Time points (0-1)
    values: [0, 1.0, 0]                       # Values at each segment
```

### Exponential Interpolation
```yaml
density_guidance:
  resampling_weight:
    breakpoints: [0.125, 0.25, 0.375]
    values:
      - 0.01
      - start: 0.01                           # Exponential interpolation
        end: 50
        alpha: 150
        start_t: 0.125
        end_t: 0.25
      - 50
      - 0.0
```

### Resolution Scheduling
```yaml
density_guidance:
  resolution:
    breakpoints: [0.25]                       # Change resolution at 25% complete
    values: [2.0, 8.0]                        # Start at 2.0Å, change to 8.0Å
```

### Advanced Scheduling Features
- **Breakpoints**: Time points between 0-1 representing fraction of optimization complete
- **Values**: Parameter values at each segment (length = breakpoints + 1)
- **Exponential interpolation**: Smooth exponential transitions between values
- **Mixed schedules**: Combine constant values with exponential interpolation
- **Resolution adaptation**: Dynamic resolution changes during optimization

## Multi-GPU Parallel Execution

The configuration system supports running multiple experiments in parallel across GPUs:

### Basic Parallel Execution
```bash
# Run 4 experiments in parallel across 4 GPUs
edg --config configs/batch_experiment.yaml --max-parallel 4

# Run 2 experiments in parallel across 2 GPUs
edg --config configs/synthetic_density_test.yaml --max-parallel 2
```

### Parallel Execution Features
- **GPU Model Management**: Boltz models loaded once per GPU and shared across experiments
- **Thread-Safe Operations**: Safe concurrent access to GPU resources
- **Error Recovery**: Automatic recovery from CUDA out-of-memory errors
- **Progress Tracking**: Real-time progress monitoring with GPU information
- **Batch Summaries**: Detailed execution summaries with success/failure statistics

### Testing Parallel Execution
```bash
# Test parallel execution functionality
python test_parallel_execution.py
```

### Parameter Sweeps with Parallel Execution
```bash
# Learning rate sweep with parallel execution
for lr in 0.01 0.02 0.05; do
  edg --config configs/batch_sweep.yaml \
      --adaptive-solver.learning-rate $lr \
      --max-parallel 2 \
      --output-base-dir "results/lr_sweep_${lr}"
done
```

## Command-Line Overrides

Any parameter can be overridden from the command line:

### Simple Parameters
```bash
edg --config my_experiment.yaml --num-steps 300 --resolution 1.5
```

### Nested Parameters
```bash
edg --config my_experiment.yaml --density-guidance.base-weight 0.8
edg --config my_experiment.yaml --adaptive-solver.learning-rate 0.02
```

### Boolean Flags
```bash
edg --config my_experiment.yaml --substructure-enabled
edg --config my_experiment.yaml --continue-on-error
```

### Batch Parameters
```bash
edg --config my_batch.yaml --max-parallel 4 --continue-on-error
```

### Complex Overrides
```bash
edg --config my_experiment.yaml --override "density.resolution=1.5" --override "model.version=boltz1"
```

## Migration from run_*.py Scripts

To convert existing `run_*.py` scripts to YAML configs:

1. **Find the original script**: e.g., `_notebooks/run_ptp1b.py`
2. **Copy a template**: `cp configs/templates/basic_experiment.yaml configs/ptp1b.yaml`
3. **Extract parameters**: Copy hardcoded values from the script to the YAML
4. **Test the conversion**: `edg --config configs/ptp1b.yaml --validate-only`
5. **Run the experiment**: `edg --config configs/ptp1b.yaml`

### Common Parameter Mappings

| Script Parameter | YAML Path |
|------------------|-----------|
| `num_steps = 200` | `diffusion.num_steps: 200` |
| `ensemble_size = 4` | `optimization.ensemble_size: 4` |
| `resolution = 2.0` | `density.resolution: 2.0` |
| `steering_args.num_particles = 3` | `steering.num_particles: 3` |
| `adaptive_solver_config.learning_rate = 0.3` | `adaptive_solver.learning_rate: 0.3` |

## Examples

### Simple Experiment
```bash
# Quick test with minimal parameters
edg --config configs/templates/simple_experiment.yaml --num-steps 10
```

### Production Run
```bash
# Full PTP1B experiment with parameter sweeps
for lr in 0.01 0.02 0.05; do
  edg --config configs/ptp1b.yaml --learning-rate $lr --output-dir "results/ptp1b_lr${lr}"
done
```

### Parameter Validation
```bash
# Check configuration before running
edg --config configs/my_experiment.yaml --validate-only
```

## Tips

1. **Start Simple**: Use `configs/templates/simple_experiment.yaml` for initial testing
2. **Version Control**: Commit your YAML configs to track parameter changes
3. **Validate Early**: Use `--validate-only` to catch errors before long runs
4. **Override Flexibly**: Use CLI overrides for parameter sweeps without editing files
5. **Save Configs**: Use `--save-config` to record the final parameters used
6. **Check Outputs**: Each run saves the complete configuration to `experiment_config.yaml`

## Troubleshooting

### File Not Found Errors
- Check that all paths in the YAML are absolute or relative to the correct working directory
- Verify structure and density files exist at the specified paths

### Validation Errors
- Use `--validate-only` to see detailed error messages
- Check that resolution is specified for CCP4/MRC files
- Verify model version is "boltz1" or "boltz2"

### Schedule Parsing Errors
- Ensure schedule types are valid: "constant", "piecewise", "exponential", etc.
- Check that breakpoints are in [0, 1] range and sorted
- Verify nested schedule syntax matches the examples above
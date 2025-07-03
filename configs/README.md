# EDG Configuration System

This directory contains YAML configuration files for EDG (Ensembles from Density Generator) experiments. The new configuration system replaces the hardcoded parameters in `run_*.py` scripts with flexible, version-controlled YAML files.

## Quick Start

### Basic Usage
```bash
# Run an experiment with a config file
edg --config configs/simple_experiment.yaml

# Override parameters from command line
edg --config configs/ptp1b.yaml --num-steps 300 --resolution 1.5

# Validate configuration without running
edg --config configs/mac1.yaml --validate-only
```

### Creating Your First Experiment
1. Copy a template: `cp configs/templates/simple_experiment.yaml configs/my_experiment.yaml`
2. Edit the paths and parameters in `my_experiment.yaml`
3. Run: `edg --config configs/my_experiment.yaml`

## Directory Structure

```
configs/
├── README.md                    # This file
├── templates/                   # Template configurations
│   ├── basic_experiment.yaml    # Full-featured template with all options
│   └── simple_experiment.yaml   # Simple template with minimal settings
├── ptp1b.yaml                  # PTP1B experiment (replaces run_ptp1b.py)
├── mac1.yaml                   # MAC1 experiment (replaces run_mac1.py)
└── your_experiment.yaml        # Your custom experiments
```

## Configuration Sections

### Required Sections

#### Experiment Identity
```yaml
name: "my_experiment"          # Experiment name for outputs
```

#### Structure Configuration
```yaml
structure:
  structure_path: "/path/to/structure.cif"    # Input structure file
  clean_structure: true                       # Apply structure cleaning
  keep_type: "protein"                        # "protein", "all"
  remove_alternative_conformations: true      # Remove alt conformations
  complete_residues: true                     # Complete missing atoms
```

#### Density Configuration
```yaml
density:
  map_path: "/path/to/density.ccp4"          # Density map file
  resolution: 2.0                             # Resolution (required for CCP4/MRC)
  em_mode: false                              # Use EM scattering factors
```

#### Output Configuration
```yaml
output_dir: "results/my_experiment"          # Where results are saved
input_data_dir: "input/my_experiment"        # Temporary input files
```

### Optional Sections (with defaults)

#### Model Configuration
```yaml
model:
  version: "boltz2"                           # "boltz1" or "boltz2"
  checkpoint_path: null                       # Auto-detect if null
  device: null                                # Auto-detect if null
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
    type: "piecewise"
    breakpoints: [0.125, 0.375]               # Time points (0-1)
    values: [0, 1.0, 0]                       # Values at each segment
```

### Exponential Interpolation
```yaml
density_guidance:
  resampling_weight:
    type: "exponential_bounds"
    start: 0.01
    end: 50
    alpha: 150
    start_t: 0.125
    end_t: 0.25
```

### Nested Schedules
```yaml
density_guidance:
  resampling_weight:
    type: "piecewise"
    breakpoints: [0.125, 0.25, 0.375]
    values: 
      - 0.01
      - type: "exponential_bounds"
        start: 0.01
        end: 50
        alpha: 150
        start_t: 0.125
        end_t: 0.25
      - 50
      - 0.0
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
```

### Boolean Flags
```bash
edg --config my_experiment.yaml --substructure-enabled
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
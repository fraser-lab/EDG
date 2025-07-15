# Scripts Directory

This directory contains utility scripts for generating EDG experiment configurations.

## generate_altloc_batch_config.py

Generates batch configuration YAML files for altloc substructure conditioning experiments from CSV data.

### Usage

```bash
python scripts/generate_altloc_batch_config.py \
  --csv tests/resources/altloc_data/mmcif_files/altloc_summary.csv \
  --template configs/ptp1b_control.yaml \
  --output configs/synthetic_density_test.yaml
```

### Parameters

- `--csv`: Path to CSV file containing altloc data with columns:
  - `pdb_code`: PDB code (e.g., "8OSW")
  - `target_chain`: Chain identifier (e.g., "A")
  - `segment_start`: Start residue number (e.g., 109)
  - `segment_end`: End residue number (e.g., 114)

- `--template`: Path to base template configuration YAML file (e.g., `configs/ptp1b_control.yaml`)

- `--output`: Path to output batch configuration YAML file

- `--structure-base-path`: Base path for structure files (default: `tests/resources/altloc_data/mmcif_files`)

- `--output-base-dir`: Base output directory (default: `results/synthetic_density_test`)

- `--input-base-dir`: Base input directory (default: `input/synthetic_density_test`)

- `--batch-name`: Name for the batch experiment (default: `synthetic_density_test`)

### Generated Configuration

The script generates a batch configuration using the `shared_config` approach with:

1. **Shared configuration** containing base parameters from the template with `null` values for fields that get overridden
2. **Individual experiment configs** with only the necessary overrides:
   - **Unique substructure selections** for each protein: `chain {target_chain} and resi {segment_start}-{segment_end}`
   - **Structure paths** pointing to `{structure_base_path}/{pdb_code}.cif`
   - **Density map paths** using synthetic density output when available
   - **Individual output directories** for each experiment

This approach dramatically reduces configuration redundancy while maintaining full functionality.

### Example Output

```yaml
name: synthetic_density_test
output_base_dir: results/synthetic_density_test
input_base_dir: input/synthetic_density_test
continue_on_error: true
max_parallel: 1

# Shared configuration with null values for overridden fields
shared_config:
  name: null                    # Will be overridden with protein name
  output_dir: null              # Will be overridden with protein-specific path
  input_data_dir: null          # Will be overridden with protein-specific path
  structure:
    structure_path: null        # Will be overridden with protein-specific path
    clean_structure: true
    keep_type: protein
    remove_all_ligands: true
    # ... other structure settings from template
  density:
    map_path: null              # Will be overridden with protein-specific path
    resolution: 2.0
    em_mode: false
  # ... all other settings from template (diffusion, steering, etc.)
  substructure:
    enabled: true
    guidance_weight: 0.05
    # ... other substructure settings from template

# Individual experiments with minimal overrides
experiments:
  - name: 8osw
    structure:
      structure_path: tests/resources/altloc_data/mmcif_files/8osw.cif
    density:
      map_path: synthetic_density_output/8osw/8osw_main_2.0A.ccp4
      resolution: 2.0
    output_dir: results/synthetic_density_test/8osw
    input_data_dir: input/synthetic_density_test/8osw
    substructure:
      enabled: true
      selection: chain A and resi 109-114
  - name: 8owz
    structure:
      structure_path: tests/resources/altloc_data/mmcif_files/8owz.cif
    density:
      map_path: synthetic_density_output/8owz/8owz_main_2.0A.ccp4
      resolution: 2.0
    output_dir: results/synthetic_density_test/8owz
    input_data_dir: input/synthetic_density_test/8owz
    substructure:
      enabled: true
      selection: chain B and resi 120-125
```

### Running the Generated Configuration

```bash
# Validate the configuration
pixi run python -m edg --config configs/synthetic_density_test.yaml --validate-only

# Run the batch experiments
pixi run python -m edg --config configs/synthetic_density_test.yaml

# Run with parameter overrides
pixi run python -m edg --config configs/synthetic_density_test.yaml --num-steps 100
```

### Key Benefits of the Shared Config Approach

- **Reduced redundancy**: Configuration size reduced by ~95% compared to individual full configs
- **Improved maintainability**: Changes to base parameters only need to be made in `shared_config`
- **Better readability**: Individual experiments show only the parameters that differ
- **Null value handling**: Fields marked as `null` in `shared_config` are required to be overridden in individual experiments
- **Validation support**: The system validates that required fields are properly overridden

### Notes

- The script automatically uses synthetic density output files when available
- All experiments use the same base parameters from the template via `shared_config`
- Each experiment gets a unique substructure selection based on CSV data
- The generated configuration includes proper parameter scheduling from the template
- Use `continue_on_error: true` to handle any failures gracefully
- Fields marked as `null` in `shared_config` must be defined in individual experiments
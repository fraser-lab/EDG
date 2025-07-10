#!/bin/bash

# Parameter sweep script for EDG experiments with shared input optimization
# 
# This script runs parameter sweeps over:
# - Adaptive solver learning rates
# - Max guidance denoising ratios  
# - Number of diffusion steps
#
# Usage: ./parameter_sweep.sh <config_file> [--dry-run]
# 
# Examples:
#   ./parameter_sweep.sh configs/ptp1b.yaml                    # Run full sweep with shared processing
#   ./parameter_sweep.sh configs/ptp1b.yaml --dry-run          # Validate only
#
# Features:
# - Shared input directories to avoid redundant Boltz processing
# - Uses new direct CLI parameters (--learning-rate, --max-guidance-denoising-ratio, --num-steps)
# - Supports dry-run mode for validation
# - Creates organized output directories with logs
# - Generates summary reports
# - Significant speedup for parameter sweeps (60-80% faster after first run)

set -e

# Parse command line arguments
DRY_RUN=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
            else
                echo "Too many arguments"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if config file is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [--dry-run]"
    echo "Example: $0 configs/ptp1b.yaml"
    echo "Example: $0 configs/ptp1b.yaml --dry-run"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Extract base name for output directory structure
BASE_NAME=$(basename "$CONFIG_FILE" .yaml)

# Parameter arrays - customize these values as needed
LEARNING_RATES=(0.0001 0.001 0.01 0.1 0.5)           # Adaptive solver learning rates
MAX_GUIDANCE_RATIOS=(0.1 0.5 1.0)         # Maximum guidance to denoising ratios
NUM_STEPS=(200 )                       # Number of diffusion steps

echo "Starting parameter sweep for $CONFIG_FILE"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - will only validate configurations"
fi
echo "Learning rates: ${LEARNING_RATES[*]}"
echo "Max guidance denoising ratios: ${MAX_GUIDANCE_RATIOS[*]}"
echo "Number of steps: ${NUM_STEPS[*]}"
echo "Total combinations: $((${#LEARNING_RATES[@]} * ${#MAX_GUIDANCE_RATIOS[@]} * ${#NUM_STEPS[@]}))"
echo ""

# Create results directory
RESULTS_DIR="results/sweep_${BASE_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Create shared directory for Boltz processing (reused across all runs)
SHARED_OUTPUT_DIR="$RESULTS_DIR/shared_boltz_processing"

mkdir -p "$SHARED_OUTPUT_DIR"

echo "All runs will use shared directory: $SHARED_OUTPUT_DIR"
echo "Individual run results will be in subdirectories with descriptive names"

# Counter for progress tracking
total_runs=$((${#LEARNING_RATES[@]} * ${#MAX_GUIDANCE_RATIOS[@]} * ${#NUM_STEPS[@]}))
current_run=0

# Log file for the sweep
LOG_FILE="$RESULTS_DIR/sweep_log.txt"
echo "Parameter sweep started at $(date)" > "$LOG_FILE"
echo "Config file: $CONFIG_FILE" >> "$LOG_FILE"
echo "Results directory: $RESULTS_DIR" >> "$LOG_FILE"
echo "Shared Boltz processing directory: $SHARED_OUTPUT_DIR" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "Note: Boltz will automatically reuse processed data in the shared directory for faster subsequent runs"

# Run parameter sweep
for lr in "${LEARNING_RATES[@]}"; do
    for ratio in "${MAX_GUIDANCE_RATIOS[@]}"; do
        for steps in "${NUM_STEPS[@]}"; do
            current_run=$((current_run + 1))
            
            echo "[$current_run/$total_runs] Running: lr=$lr, ratio=$ratio, steps=$steps"
            
            # Log the run
            echo "Run $current_run/$total_runs: lr=$lr, ratio=$ratio, steps=$steps" >> "$LOG_FILE"
            echo "Started at: $(date)" >> "$LOG_FILE"
            
            # Build command - all runs use the same output directory for Boltz processing reuse
            # The experiment runner will create unique subdirectories based on parameters
            cmd="pixi run python -m edg --config \"$CONFIG_FILE\" --learning-rate $lr --max-guidance-denoising-ratio $ratio --num-steps $steps --output-dir \"$SHARED_OUTPUT_DIR\""
            
            # Add validation-only flag for dry run
            if [ "$DRY_RUN" = true ]; then
                cmd="$cmd --validate-only"
            fi
            
            # Run the experiment and log output to a temporary file
            temp_log="$RESULTS_DIR/temp_lr${lr}_ratio${ratio}_steps${steps}.log"
            
            if eval "$cmd 2>&1 | tee \"$temp_log\""; then
                
                if [ "$DRY_RUN" = true ]; then
                    echo "✓ Configuration validated" >> "$LOG_FILE"
                    echo "✓ Configuration validated"
                else
                    echo "✓ Completed successfully" >> "$LOG_FILE"
                    echo "✓ Run completed successfully"
                    echo "Results saved to automatically created subdirectory in: $SHARED_OUTPUT_DIR"
                fi
                
                # Move log to permanent location in results directory
                mv "$temp_log" "$RESULTS_DIR/run_lr${lr}_ratio${ratio}_steps${steps}.log"
                
            else
                echo "✗ Failed with exit code $?" >> "$LOG_FILE"
                echo "✗ Run failed - check $RESULTS_DIR/run_lr${lr}_ratio${ratio}_steps${steps}.log for details"
                mv "$temp_log" "$RESULTS_DIR/run_lr${lr}_ratio${ratio}_steps${steps}.log"
            fi
            
            echo "Finished at: $(date)" >> "$LOG_FILE"
            echo "" >> "$LOG_FILE"
            echo ""
        done
    done
done

echo "Parameter sweep completed!"
echo "Results saved in: $RESULTS_DIR"
echo "Summary log: $LOG_FILE"

# Create a summary of all runs
SUMMARY_FILE="$RESULTS_DIR/sweep_summary.txt"
echo "Parameter Sweep Summary" > "$SUMMARY_FILE"
echo "======================" >> "$SUMMARY_FILE"
echo "Config file: $CONFIG_FILE" >> "$SUMMARY_FILE"
echo "Total runs: $total_runs" >> "$SUMMARY_FILE"
echo "Started: $(head -1 "$LOG_FILE" | cut -d' ' -f4-)" >> "$SUMMARY_FILE"
echo "Completed: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Count successful runs
successful_runs=$(grep -c "✓ Completed successfully" "$LOG_FILE" || echo "0")
echo "Successful runs: $successful_runs/$total_runs" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Individual run logs:" >> "$SUMMARY_FILE"
ls -1 "$RESULTS_DIR" | grep -E "^run_lr.*_ratio.*_steps.*\.log$" | sort >> "$SUMMARY_FILE"

echo "" >> "$SUMMARY_FILE"
echo "Experiment results:" >> "$SUMMARY_FILE"
echo "  Shared Boltz processing directory: $SHARED_OUTPUT_DIR" >> "$SUMMARY_FILE"
echo "  Individual results in subdirectories: $(ls -1 "$SHARED_OUTPUT_DIR" | grep -E "^boltz" | wc -l) subdirectories created" >> "$SUMMARY_FILE"
echo "  Note: Each parameter combination creates a unique subdirectory based on its parameters" >> "$SUMMARY_FILE"

echo ""
echo "Summary saved in: $SUMMARY_FILE"
#!/usr/bin/env python
"""
Calculate R-work and R-free between two MTZ files.
The first MTZ file is treated as the "experimental" reference and provides R-free flags.
The second MTZ file contains F values to compare against.
"""

import subprocess
import numpy as np
import sys
import os

def run_phenix_mtz_dump(mtz_file):
    """Extract data from MTZ using phenix.mtz.dump"""
    cmd = ["phenix.mtz.dump", mtz_file, "-c", "-f", "machine_readable"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.stderr:
        print(f"Warning from phenix.mtz.dump: {result.stderr}", file=sys.stderr)
    return result.stdout

def parse_mtz_dump(dump_output, f_label="F", rfree_label="R-free-flags", extract_rfree=True):
    """
    Parse phenix.mtz.dump output in 'machine_readable' format
    to extract HKL, F, and optionally R-free flags.
    """
    lines = dump_output.strip().split('\n')
    
    # Find column labels and their order
    column_labels = []
    
    # Locate the start of column labels
    try:
        start_labels_idx = lines.index("Column labels (one per line):")
    except ValueError:
        print("Error: 'Column labels (one per line):' not found in dump output.", file=sys.stderr)
        return {}, {}
        
    num_columns_line_idx = start_labels_idx - 1
    if num_columns_line_idx < 0 or not lines[num_columns_line_idx].startswith("Number of columns:"):
        print("Error: Could not find 'Number of columns:' line before column labels.", file=sys.stderr)
        return {}, {}
    
    num_columns = int(lines[num_columns_line_idx].split(':')[-1].strip())
    
    # Read column labels
    for i in range(num_columns):
        if start_labels_idx + 1 + i < len(lines):
            column_labels.append(lines[start_labels_idx + 1 + i].strip())
        else:
            print("Error: Not enough column labels found in dump output.", file=sys.stderr)
            return {}, {}

    # Find relevant column indices
    f_val_idx = None
    rfree_val_idx = None

    # Try to find F column - might have different labels
    possible_f_labels = [f_label, "FMODEL", "FC", "FCALC", "F_model", "F_calc"]
    for label in possible_f_labels:
        if label in column_labels:
            f_val_idx = column_labels.index(label)
            print(f"Found F column with label: {label}")
            break
    
    if f_val_idx is None:
        print(f"Error: No F column found. Tried labels: {possible_f_labels}", file=sys.stderr)
        print(f"Available columns: {column_labels}", file=sys.stderr)
        return {}, {}

    # Look for R-free flags if requested
    if extract_rfree:
        try:
            rfree_val_idx = column_labels.index(rfree_label)
        except ValueError:
            print(f"Warning: R-free label '{rfree_label}' not found. Will return empty rfree_flags.", file=sys.stderr)
            rfree_val_idx = -1

    # Locate the start of reflection data
    try:
        start_data_idx = lines.index("Column data (HKL followed by data):")
    except ValueError:
        print("Error: 'Column data (HKL followed by data):' not found in dump output.", file=sys.stderr)
        return {}, {}

    reflections = {}
    rfree_flags = {}
    
    current_line_idx = start_data_idx + 1
    while current_line_idx < len(lines):
        hkl_line = lines[current_line_idx].strip()
        try:
            h, k, l = map(int, hkl_line.split())
            current_line_idx += 1
        except ValueError:
            # Not an HKL line, probably end of data
            break

        data_vals = []
        for _ in range(num_columns):
            if current_line_idx < len(lines):
                try:
                    data_vals.append(float(lines[current_line_idx].strip()))
                    current_line_idx += 1
                except ValueError:
                    print(f"Warning: Could not parse data value at line {current_line_idx+1}. Skipping reflection.", file=sys.stderr)
                    current_line_idx += (num_columns - len(data_vals))
                    break
            else:
                print(f"Error: Unexpected end of file while reading data for HKL ({h},{k},{l}).", file=sys.stderr)
                break

        if len(data_vals) == num_columns:
            f_val = data_vals[f_val_idx]
            reflections[(h, k, l)] = f_val
            
            if extract_rfree and rfree_val_idx != -1:
                rfree = int(data_vals[rfree_val_idx])
                rfree_flags[(h, k, l)] = rfree
            
    return reflections, rfree_flags

def calculate_r_factors(f_obs_dict, f_calc_dict, rfree_flags):
    """Calculate R-work and R-free"""
    
    # Find common reflections
    common_hkl = set(f_obs_dict.keys()) & set(f_calc_dict.keys())
    
    if not common_hkl:
        print("Error: No common reflections found between the two MTZ files.", file=sys.stderr)
        return None, None, 0, 0
    
    print(f"\nFound {len(common_hkl)} common reflections out of:")
    print(f"  MTZ1: {len(f_obs_dict)} reflections")
    print(f"  MTZ2: {len(f_calc_dict)} reflections")
    
    # Separate work and test sets
    if rfree_flags:
        work_hkl = [hkl for hkl in common_hkl if rfree_flags.get(hkl, 0) == 0]
        test_hkl = [hkl for hkl in common_hkl if rfree_flags.get(hkl, 0) == 1]
    else:
        print("Warning: No R-free flags found. Treating all reflections as work set.", file=sys.stderr)
        work_hkl = list(common_hkl)
        test_hkl = []
    
    def calc_r(hkl_list, label):
        if not hkl_list:
            print(f"\n{label}: No reflections in this set.", file=sys.stderr)
            return None, 0
            
        f_obs_vals = np.array([f_obs_dict[hkl] for hkl in hkl_list])
        f_calc_vals = np.array([f_calc_dict[hkl] for hkl in hkl_list])
        
        # Calculate unscaled R-factor
        sum_abs_diff_unscaled = np.sum(np.abs(f_obs_vals - f_calc_vals))
        sum_abs_fobs = np.sum(np.abs(f_obs_vals))
        r_unscaled = sum_abs_diff_unscaled / sum_abs_fobs if sum_abs_fobs > 0 else 0
        
        # Calculate scale factor
        fc_dot_fc = np.dot(f_calc_vals, f_calc_vals)
        scale = np.dot(f_obs_vals, f_calc_vals) / fc_dot_fc if fc_dot_fc > 0 else 1.0
        
        # Apply scale to F_calc
        scaled_f_calc_vals = scale * f_calc_vals
        
        # Calculate scaled R-factor
        sum_abs_diff = np.sum(np.abs(f_obs_vals - scaled_f_calc_vals))
        r_scaled = sum_abs_diff / sum_abs_fobs if sum_abs_fobs > 0 else 0
        
        # Calculate correlation coefficient
        cc = np.corrcoef(f_obs_vals, f_calc_vals)[0, 1]
        
        print(f"\n{label}:")
        print(f"  Reflections: {len(hkl_list)}")
        print(f"  R-factor (unscaled): {r_unscaled:.4f} ({r_unscaled*100:.2f}%)")
        print(f"  R-factor (scaled): {r_scaled:.4f} ({r_scaled*100:.2f}%)")
        print(f"  Scale factor: {scale:.3f}")
        print(f"  Correlation coefficient: {cc:.4f}")
        print(f"  <F_obs>: {np.mean(f_obs_vals):.2f}")
        print(f"  <F_calc>: {np.mean(f_calc_vals):.2f}")
        
        return r_scaled, len(hkl_list)
        
    r_work, n_work = calc_r(work_hkl, "R-work")
    r_free, n_test = calc_r(test_hkl, "R-free") if test_hkl else (None, 0)
    
    return r_work, r_free, n_work, n_test

def main():
    if len(sys.argv) < 3:
        print("Usage: phenix.python rfree_between_mtz.py experiment.mtz comparison.mtz [f_label_exp] [f_label_comp]")
        print("\nCalculates R-work and R-free between two MTZ files.")
        print("  experiment.mtz  - MTZ file with 'experimental' F values and R-free flags")
        print("  comparison.mtz  - MTZ file with F values to compare against")
        print("  f_label_exp     - F column label in experiment.mtz (default: F)")
        print("  f_label_comp    - F column label in comparison.mtz (default: searches common labels)")
        print("\nExample: phenix.python rfree_between_mtz.py rfree_2A_Waltconf.mtz model_fc.mtz F FMODEL")
        sys.exit(1)
        
    exp_mtz = sys.argv[1]
    comp_mtz = sys.argv[2]
    f_label_exp = sys.argv[3] if len(sys.argv) > 3 else "F"
    f_label_comp = sys.argv[4] if len(sys.argv) > 4 else "F"
    
    if not os.path.exists(exp_mtz):
        print(f"Error: Experimental MTZ file '{exp_mtz}' not found.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(comp_mtz):
        print(f"Error: Comparison MTZ file '{comp_mtz}' not found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("R-free Calculation Between Two MTZ Files")
    print("=" * 60)
    print(f"Experimental MTZ: {exp_mtz}")
    print(f"Comparison MTZ: {comp_mtz}")
    print(f"F label (exp): {f_label_exp}")
    print(f"F label (comp): {f_label_comp}")
    
    # Extract F and R-free from experimental MTZ
    print("\nExtracting data from experimental MTZ...")
    try:
        dump_exp = run_phenix_mtz_dump(exp_mtz)
        f_obs, rfree_flags = parse_mtz_dump(dump_exp, f_label=f_label_exp, extract_rfree=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running phenix.mtz.dump on {exp_mtz}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing experimental MTZ: {e}", file=sys.stderr)
        sys.exit(1)

    if not f_obs:
        print("No F_obs reflections found in experimental MTZ. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(f_obs)} reflections in experimental MTZ")
    if rfree_flags:
        n_work = sum(1 for flag in rfree_flags.values() if flag == 0)
        n_test = sum(1 for flag in rfree_flags.values() if flag == 1)
        print(f"R-free flags: {n_work} work, {n_test} test")
    
    # Extract F from comparison MTZ
    print("\nExtracting data from comparison MTZ...")
    try:
        dump_comp = run_phenix_mtz_dump(comp_mtz)
        f_calc, _ = parse_mtz_dump(dump_comp, f_label=f_label_comp, extract_rfree=False)
    except subprocess.CalledProcessError as e:
        print(f"Error running phenix.mtz.dump on {comp_mtz}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing comparison MTZ: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not f_calc:
        print("No F reflections found in comparison MTZ. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(f_calc)} reflections in comparison MTZ")
    
    # Calculate R-factors
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    r_work, r_free, n_work, n_test = calculate_r_factors(f_obs, f_calc, rfree_flags)
    
    if r_work is not None:
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print(f"R-work: {r_work:.4f} ({r_work*100:.2f}%)")
        if r_free is not None:
            print(f"R-free: {r_free:.4f} ({r_free*100:.2f}%)")
            print(f"R-free - R-work: {r_free - r_work:.4f}")
            print(f"Work/Test ratio: {n_work}/{n_test}")
        else:
            print("R-free: Not calculated (no test reflections)")
    else:
        print("Could not calculate R-factors due to insufficient data.", file=sys.stderr)

if __name__ == "__main__":
    main()
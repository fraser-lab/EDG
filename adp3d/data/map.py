"""Preprocess MTZ into maps for utilization with ADP3D."""

# James Holton's advice on how to make these CryoEM-similar maps

# expand to P1
# calculate the mFo-DFc difference map
# calculate an Fcalc map for just the protomer of interest -> done in PHENIX
# make your "carving" mask around the protomer of interest
# I prefer to make a "feathered" mask, dropping from "1" to "0" at non-bonding distances, say 3.5 to 4.5 A - I have a script for this
# apply the mask to the mFo-DFc map
# add the "carved" Fo-Fc map to the Fcalc map. this effectively subtracts the symmetry mates

import subprocess
import os
import gemmi
import numpy as np

TC_MASKIFY_PATH = "adp3d/utils/_jh_scripts/Tc_maskify.com"
MAP_FUNC_PATH = "adp3d/utils/_jh_scripts/map_func.com"
PHENIX_REFINE_PATH = "adp3d/utils/single_refinement.sh"
TMP_DIR = "tmp"

def run_phenix_refine(pdb_file, mtz_file):
    cmd = ["bash", PHENIX_REFINE_PATH, mtz_file, pdb_file]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e.stderr}")


def run_tc_maskify(pdb_file, map_file, tc_maskify_path, map_func_path, b_factor=50, tc_scale=1, output_file="mask.map"):
   # add map_func.com directory to PATH
   os.environ['PATH'] = f"{os.path.dirname(map_func_path)}:{os.environ['PATH']}"
   
   os.chmod(tc_maskify_path, 0o755)
   os.chmod(map_func_path, 0o755)
   
   cmd = [
       tc_maskify_path,
       f"pdbfile={pdb_file}",
       f"mapfile={map_file}", 
       f"B={b_factor}",
       f"Tc_scale={tc_scale}",
       f"outfile={output_file}"
   ]
   
   try:
       result = subprocess.run(cmd, capture_output=True, text=True, check=True)
       print(result.stdout)
   except subprocess.CalledProcessError as e:
       print(f"Error running script: {e.stderr}")

# assuming you are running from the base directory of the github repo
run_tc_maskify("structure.pdb", "density.map", 
              "/Tc_maskify.com",
              "/path/to/map_func.com", 
              b_factor=60)
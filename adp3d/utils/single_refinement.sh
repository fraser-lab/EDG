#!/usr/bin/env bash

source /home/misc/software/phenix/phenix-1.21-5190/phenix_env.sh

# Assert required files exist
mapfile=$1
pdb=$2
pdb_name="${mapfile%.mtz}"

#__________________________________DETERMINE RESOLUTION AND (AN)ISOTROPIC REFINEMENT__________________________________
mtzmetadata=`phenix.mtz.dump "${pdb_name}.mtz"`
resrange=`grep "Resolution range:" <<< "${mtzmetadata}"`

echo "${resrange}"

res=`echo "${resrange}" | cut -d " " -f 4 | cut -c 1-5`
res1000=`echo $res | awk '{tot = $1*1000}{print tot }'`

if (( $res1000 < 1550 )); then
  adp='adp.individual.anisotropic="not (water or element H)"'
else
  adp='adp.individual.isotropic=all'
fi


#__________________________________DETERMINE FOBS v IOBS v FP__________________________________
# List of Fo types we will check for
obstypes=("FP" "FOBS" "FSIM" "F-obs" "I" "IOBS" "I-obs" "F(+)" "I(+)")

# Get amplitude fields
ampfields=`grep -E "amplitude|intensity|F\(\+\)|I\(\+\)" <<< "${mtzmetadata}"`
ampfields=`echo "${ampfields}" | awk '{$1=$1};1' | cut -d " " -f 1`

# Clear xray_data_labels variable
xray_data_labels=""

# Is amplitude an Fo?
for field in ${ampfields}; do
  # Check field in obstypes
  if [[ " ${obstypes[*]} " =~ " ${field} " ]]; then
    # Check SIGFo is in the mtz too!
    if grep -F -q -w "SIG$field" <<< "${mtzmetadata}"; then
      xray_data_labels="${field},SIG${field}";
      break
    fi
  fi
done
if [ -z "${xray_data_labels}" ]; then
  echo >&2 "Could not determine Fo field name with corresponding SIGFo in .mtz.";
  echo >&2 "Was not among "${obstypes[*]}". Please check .mtz file\!";
  exit 1;
else
  echo "data labels: ${xray_data_labels}"
fi

#_____________________________DETERMINE R FREE FLAGS______________________________
gen_Rfree=True
rfreetypes="FREE R-free-flags"
for field in ${rfreetypes}; do
  if grep -F -q -w $field <<< "${mtzmetadata}"; then
    gen_Rfree=False;
    echo "Rfree column: ${field}";
    r_free_flags="${field}";
    break
  fi
done
echo "data_manager {" > ${pdb_name}_single_refine.params
echo "    miller_array {" >> ${pdb_name}_single_refine.params
echo "        labels {" >> ${pdb_name}_single_refine.params
echo "            name = \"$xray_data_labels\"" >> ${pdb_name}_single_refine.params
echo "            type = x_ray" >> ${pdb_name}_single_refine.params
echo "        }" >> ${pdb_name}_single_refine.params
if [ "${gen_Rfree}" = "False" ]; then
    echo "        labels {" >> ${pdb_name}_single_refine.params
    echo "            name = \"$r_free_flags\"" >> ${pdb_name}_single_refine.params
    echo "            type = x_ray" >> ${pdb_name}_single_refine.params
    echo "        }" >> ${pdb_name}_single_refine.params
fi
echo "    }" >> ${pdb_name}_single_refine.params
echo "    fmodel {" >> ${pdb_name}_single_refine.params
echo "        xray_data {" >> ${pdb_name}_single_refine.params
echo "            r_free_flags {" >> ${pdb_name}_single_refine.params
echo "                generate = ${gen_Rfree}" >> ${pdb_name}_single_refine.params
echo "            }" >> ${pdb_name}_single_refine.params
echo "        }" >> ${pdb_name}_single_refine.params
echo "    }" >> ${pdb_name}_single_refine.params
echo "}" >> ${pdb_name}_single_refine.params


#__________________________________FINAL REFINEMENT__________________________________
# Write refinement parameters into parameters file
echo "refinement.refine.strategy=*individual_sites *individual_adp *occupancies"  >> ${pdb_name}_single_refine.params
echo "output.prefix=${pdb_name}_single" >> ${pdb_name}_single_refine.params
echo "refinement.main.number_of_macro_cycles=5"  >> ${pdb_name}_single_refine.params
echo "refinement.main.nqh_flips=True"            >> ${pdb_name}_single_refine.params
echo "refinement.refine.${adp}"                  >> ${pdb_name}_single_refine.params
echo "refinement.output.write_maps=False"        >> ${pdb_name}_single_refine.params
echo "refinement.hydrogens.refine=riding"        >> ${pdb_name}_single_refine.params
echo "refinement.main.ordered_solvent=True"      >> ${pdb_name}_single_refine.params
echo "refinement.main.nproc=4"                  >> ${pdb_name}_single_refine.params
echo "refinement.main.optimize_mask=True"        >> ${pdb_name}_single_refine.params
echo "refinement.target_weights.optimize_xyz_weight=true"  >> ${pdb_name}_single_refine.params
echo "refinement.target_weights.optimize_adp_weight=true"  >> ${pdb_name}_single_refine.params

phenix.refine "${pdb}" "${pdb_name}.mtz" "${pdb_name}_single_refine.params" --overwrite

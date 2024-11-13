"""Preprocess maps for utilization with ADP3D."""

# TODO: Follow James Holton's advice on how to make these CryoEM-similar maps

# expand to P1
# calculate the mFo-DFc difference map
# calculate an Fcalc map for just the protomer of interest
# make your "carving" mask around the protomer of interest
# I prefer to make a "feathered" mask, dropping from "1" to "0" at non-bonding distances, say 3.5 to 4.5 A - I have a script for this
# apply the mask to the mFo-DFc map
# add the "carved" Fo-Fc map to the Fcalc map. this effectively subtracts the symmetry mates
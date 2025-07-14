import numpy as np
import copy
import math
import logging
import itertools as itl
from .base_structure import _BaseStructure
from .math import *
from .rotamers import ROTAMERS
import time


logger = logging.getLogger(__name__)


_PROTEIN_BACKBONE_ATOMS = ["N", "CA", "C"]
_NUCLEOTIDE_BACKBONE_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3"]
_SOLVENTS = ["HOH"]


def residue_type(residue):
    """Check residue type.

    The following types are recognized:
        rotamer-residue
            Amino acid for which rotamer information is available
        aa-residue
            Amino acid residue
        residue
            DNA / RNA residue
        solvent
            HOH
        ligand
            Other
    """

    # RNA and DNA backbone atoms
    if residue.resn[0] in ROTAMERS:
        return "rotamer-residue"
    bb_atoms = _PROTEIN_BACKBONE_ATOMS
    if residue.select("name", bb_atoms).size == 3:
        return "aa-residue"
    xna_atoms = _NUCLEOTIDE_BACKBONE_ATOMS
    if residue.select("name", xna_atoms).size == 6:
        return "residue"
    if residue.resn[0] in _SOLVENTS:
        return "solvent"
    return "ligand"


class _BaseResidue(_BaseStructure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = (kwargs["resi"], kwargs["icode"])
        self.type = kwargs["type"]

    def __repr__(self):
        resi, icode = self.id
        string = "Residue: {}".format(resi)
        if self.id[1] != "":
            string += ":{}".format(icode)
        return string

    @property
    def _identifier_tuple(self):
        """Returns (chain, resi, icode) to identify this residue."""
        chainid = self.chain[0]
        resi, icode = self.id

        return (chainid, resi, icode)

    @property
    def shortcode(self):
        (chainid, resi, icode) = self._identifier_tuple
        shortcode = f"{chainid}_{resi}"
        if icode:
            shortcode += f"_{icode}"

        return shortcode

    @classmethod
    def from_residue_group(cls, residue_group, **kwargs):
        """Create a complete residue from a residue group.

        Args:
            residue_group: The residue group to create the residue from
            **kwargs: Additional arguments to pass to the residue constructor

        Returns:
            A new residue instance with proper parent data handling
        """
        # Get the selection from the residue group
        selection = residue_group._selection

        # Create a new residue with the same data and selection
        residue = cls(
            residue_group.data,
            selection=selection,
            parent=residue_group.parent,
            resi=residue_group.id[0],
            icode=residue_group.id[1],
            type=residue_type(residue_group),
            **kwargs,
        )

        return residue


class _Residue(_BaseResidue):
    pass


class _RotamerResidue(_BaseResidue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resname = self.resn[0]
        self._rotamers = ROTAMERS[resname]
        self.nchi = self._rotamers["nchi"]
        self.nrotamers = len(self._rotamers["rotamers"])
        self.rotamers = self._rotamers["rotamers"]
        self._init_clash_detection()
        self.i = 0

    def _init_clash_detection(self, scaling_factor=0.75, covalent_bonds=None):
        # Setup the condensed distance based arrays for clash detection and fill them
        self._ndistances = self.natoms * (self.natoms - 1) // 2
        self._clash_mask = np.ones(self._ndistances, bool)
        self._clash_radius2 = np.zeros(self._ndistances, float)
        radii = self.covalent_radius
        bonds = self._rotamers["bonds"]
        if covalent_bonds:
            for bond in covalent_bonds:
                bonds.append(bond)
        offset = self.natoms * (self.natoms - 1) // 2
        for i in range(self.natoms - 1):
            # starting_index = int(offset - sp_comb(self.natoms - i, 2)) - i - 1
            natoms_left = self.natoms - i
            starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i - 1
            name1 = self.name[i]
            covrad1 = radii[i]
            for j in range(i + 1, self.natoms):
                bond1 = [name1, self.name[j]]
                bond2 = bond1[::-1]
                covrad2 = radii[j]
                index = starting_index + j
                self._clash_radius2[index] = covrad1 + covrad2 + 0.4
                if bond1 in bonds or bond2 in bonds:
                    self._clash_mask[index] = False

        self._clash_radius2 *= self._clash_radius2
        self._clash_radius2 *= scaling_factor
        self._clashing = np.zeros(self._ndistances, bool)
        self._dist2_matrix = np.empty(self._ndistances, float)

        # Check if parent has active state, otherwise initialize all as active
        if self.parent is not None and "active" in self.parent.data:
            self.active = self.parent.data["active"][self._selection].copy()
        else:
            self.active = np.ones(self.natoms, dtype=bool)

        # Initialize the active mask based on atom active states
        self._active_mask = np.ones(self._ndistances, bool)
        self.update_clash_mask()

    def update_clash_mask(self):
        offset = self.natoms * (self.natoms - 1) // 2 - 1
        for i, active in enumerate(self.active[:-1]):
            natoms_left = self.natoms - i
            starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i
            end = starting_index + self.natoms - i + 1
            self._active_mask[starting_index:end] = active

    def clashes(self):
        """Checks if there are any internal clashes.
        Deactivated atoms are not taken into account.
        """

        dm = self._dist2_matrix
        coor = self.coor
        dot = np.dot
        k = 0
        for i in range(self.natoms - 1):
            u = coor[i]
            for j in range(i + 1, self.natoms):
                u_v = u - coor[j]
                dm[k] = dot(u_v, u_v)
                k += 1

        np.less_equal(dm, self._clash_radius2, self._clashing)
        self._clashing &= self._clash_mask
        self._clashing &= self._active_mask
        nclashes = self._clashing.sum()
        return nclashes

    def get_chi(self, chi_index):
        atoms = self._rotamers["chi"][chi_index]
        selection = self.select("name", atoms)
        ordered_sel = []
        for atom in atoms:
            for sel in selection:
                if atom == self._name[sel]:
                    ordered_sel.append(sel)
                    break
        coor = self._coor[ordered_sel]
        angle = dihedral_angle(coor)
        return angle

    def set_chi(self, chi_index, value, covalent=None, length=None):
        atoms = self._rotamers["chi"][chi_index]
        selection = self.select("name", atoms)

        # Translate coordinates to center on coor[1]
        coor = self._coor[selection]
        origin = coor[1].copy()
        coor -= origin

        # Make an orthogonal axis system based on 3 atoms
        backward = gram_schmidt_orthonormal_zx(coor)
        forward = backward.T

        # Complete selection to be rotated
        atoms_to_rotate = self._rotamers["chi-rotate"][chi_index]
        selection = self.select("name", atoms_to_rotate)
        if covalent in atoms_to_rotate:
            # If we are rotating the atom that is covalently bonded
            # to the ligand, we should also rotate the ligand.
            atoms_to_rotate2 = self.name[length:]
            selection2 = self.select("name", atoms_to_rotate2)
            selection = np.array(list(selection) + list(selection2), dtype=int)

        # Create transformation matrix
        angle = np.deg2rad(value - self.get_chi(chi_index))
        rotation = Rz(angle)
        R = forward @ rotation @ backward

        # Apply transformation
        coor_to_rotate = self._coor[selection]
        coor_to_rotate -= origin
        coor_to_rotate = np.dot(coor_to_rotate, R.T)
        coor_to_rotate += origin
        self._coor[selection] = coor_to_rotate

    def print_residue(self):
        for atom, coor, element, b, q in zip(
            self.name, self.coor, self.e, self.b, self.q
        ):
            logger.info(f"{atom} {coor} {element} {b} {q}")

    def _print_residue_shape(self):
        """Prints shapes of all attributes in a residue.

        NB: only of use for debugging.
        """
        logger.debug(f"[{self}]")
        for attr_name in self.REQUIRED_ATTRIBUTES:
            logger.debug(f"  {attr_name}.shape: {getattr(self, attr_name).shape}")
            logger.debug(
                f"  _{attr_name}.shape: {getattr(self, '_' + attr_name).shape}"
            )

    def complete_residue(self):
        if residue_type(self) != "rotamer-residue":
            msg = (
                f"Cannot complete non-aminoacid residue {self}. "
                f"Please complete the missing atoms of the residue before "
                f"running qFit again!"
            )
            raise RuntimeError(msg)

        failed_atoms = []
        for atom, position in zip(self._rotamers["atoms"], self._rotamers["positions"]):
            # Found a missing atom!
            if atom not in self.name:
                try:
                    self.complete_residue_recursive(atom)
                except RuntimeError as e:
                    failed_atoms.append((atom, str(e)))
                    logger.warning(f"Failed to rebuild missing atom {atom}: {e}")
            else:
                # Check if atom exists but has NaN coordinates
                atom_idx = np.argwhere(self.name == atom)[0][0]
                if np.any(np.isnan(self.coor[atom_idx])):
                    logger.info(f"Found atom {atom} with NaN coordinates - rebuilding")
                    try:
                        self.complete_residue_recursive(atom)
                    except RuntimeError as e:
                        failed_atoms.append((atom, str(e)))
                        logger.warning(f"Failed to rebuild atom {atom} with NaN coordinates: {e}")
        
        if failed_atoms:
            failed_names = [atom for atom, _ in failed_atoms]
            logger.warning(
                f"Completed residue {self} with {len(failed_atoms)} failed atoms: {failed_names}. "
                f"These atoms will be missing from the final structure."
            )

    def complete_residue_recursive(self, atom):
        if atom in ["N", "C", "CA", "O"]:
            msg = (
                f"{self} is missing backbone atom {atom}. "
                f"qFit cannot complete missing backbone atoms. "
                f"Please complete the missing backbone atoms of "
                f"the residue before running qFit again!"
            )
            raise RuntimeError(msg)

        ref_atom = self._rotamers["connectivity"][atom][0]
        if ref_atom not in self.name:
            print("ref atom:")
            print(ref_atom)
            self.complete_residue_recursive(ref_atom)
        else:
            # Check if reference atom has valid coordinates
            ref_idx = np.argwhere(self.name == ref_atom)[0][0]
            if np.any(np.isnan(self.coor[ref_idx])):
                logger.info(
                    f"Reference atom {ref_atom} has NaN coordinates - rebuilding first"
                )
                self.complete_residue_recursive(ref_atom)

        idx = np.argwhere(self.name == ref_atom)[0][0]
        ref_coor = self.coor[idx]
        bond_length, bond_length_sd = self._rotamers["bond_dist"][ref_atom][atom]
        # Identify a suitable atom for the bond angle:
        for angle in self._rotamers["bond_angle"]:
            if angle[0][1] == ref_atom and angle[0][2] == atom:
                if angle[0][0][0] == "H":
                    continue
                bond_angle_atom = angle[0][0]
                bond_angle, bond_angle_sd = angle[1]
                if bond_angle_atom not in self.name:
                    self.complete_residue_recursive(bond_angle_atom)
                else:
                    # Check if bond angle atom has valid coordinates
                    ba_idx = np.argwhere(self.name == bond_angle_atom)[0][0]
                    if np.any(np.isnan(self.coor[ba_idx])):
                        logger.info(
                            f"Bond angle atom {bond_angle_atom} has NaN coordinates - rebuilding first"
                        )
                        self.complete_residue_recursive(bond_angle_atom)

                bond_angle_coor = self.coor[
                    np.argwhere(self.name == bond_angle_atom)[0]
                ]
                dihedral_atom = None
                # If the atom's position is dependent on a rotamer,
                # identify the fourth dihedral angle atom:
                for i, chi in self._rotamers["chi"].items():
                    if (
                        chi[1] == bond_angle_atom
                        and chi[2] == ref_atom
                        and chi[3] == atom
                    ):
                        dihedral_atom = chi[0]
                        dihed_angle = self._rotamers["rotamers"][0][i - 1]

                # If the atom's position is not dependent on a rotamer,
                # identify the fourth dihedral angle atom:
                if dihedral_atom is None:
                    for dihedral in self._rotamers["dihedral"]:
                        if (
                            dihedral[0][1] == bond_angle_atom
                            and dihedral[0][2] == ref_atom
                            and dihedral[0][3] == atom
                        ):
                            dihedral_atom = dihedral[0][0]
                            if dihedral[1][0] in self._rotamers["atoms"]:
                                other_dihedral_atom = dihedral[1][0]
                                if dihedral_atom not in self.name:
                                    self.complete_residue_recursive(dihedral_atom)
                                else:
                                    # Check if dihedral atom has valid coordinates
                                    da_idx = np.argwhere(self.name == dihedral_atom)[0][
                                        0
                                    ]
                                    if np.any(np.isnan(self.coor[da_idx])):
                                        logger.info(
                                            f"Dihedral atom {dihedral_atom} has NaN coordinates - rebuilding first"
                                        )
                                        self.complete_residue_recursive(dihedral_atom)

                                dihedral_atom_coor = self.coor[
                                    np.argwhere(self.name == dihedral_atom)[0]
                                ]
                                if other_dihedral_atom not in self.name:
                                    self.complete_residue_recursive(other_dihedral_atom)
                                else:
                                    # Check if other dihedral atom has valid coordinates
                                    oda_idx = np.argwhere(
                                        self.name == other_dihedral_atom
                                    )[0][0]
                                    if np.any(np.isnan(self.coor[oda_idx])):
                                        logger.info(
                                            f"Other dihedral atom {other_dihedral_atom} has NaN coordinates - rebuilding first"
                                        )
                                        self.complete_residue_recursive(
                                            other_dihedral_atom
                                        )

                                other_dihedral_atom_coor = self.coor[
                                    np.argwhere(self.name == other_dihedral_atom)[0]
                                ]
                                try:
                                    dihed_angle = dihedral[1][1]
                                    dihed_angle += dihedral_angle(
                                        [
                                            dihedral_atom_coor[0],
                                            bond_angle_coor[0],
                                            ref_coor[0],
                                            other_dihedral_atom_coor[0],
                                        ]
                                    )
                                    if dihed_angle > 180:
                                        dihed_angle -= 360  # wrap to (-180, 180]
                                except:
                                    dihed_angle = 180
                                    dihed_angle += dihedral_angle(
                                        [
                                            dihedral_atom_coor[0],
                                            bond_angle_coor[0],
                                            ref_coor[0],
                                            other_dihedral_atom_coor[0],
                                        ]
                                    )
                                    if dihed_angle > 180:
                                        dihed_angle -= 360  # wrap to (-180, 180]
                            else:
                                dihed_angle = dihedral[1][0]
                            break
                if dihedral_atom is not None:
                    if dihedral_atom not in self.name:
                        self.complete_residue_recursive(dihedral_atom)
                    else:
                        # Check if dihedral atom has valid coordinates
                        da_idx = np.argwhere(self.name == dihedral_atom)[0][0]
                        if np.any(np.isnan(self.coor[da_idx])):
                            logger.info(
                                f"Dihedral atom {dihedral_atom} has NaN coordinates - rebuilding first"
                            )
                            self.complete_residue_recursive(dihedral_atom)

                    dihedral_atom_coor = self.coor[
                        np.argwhere(self.name == dihedral_atom)[0]
                    ]
                    break

        logger.debug(
            f"Rebuilding {atom}:\n"
            f"  {dihedral_atom}@{dihedral_atom_coor.flatten()}\n"
            f"  {bond_angle_atom}@{bond_angle_coor.flatten()}\n"
            f"  {ref_atom}@{ref_coor.flatten()}\n"
            f"  length:{bond_length}±{bond_length_sd}\n"
            f"  angle:{bond_angle}±{bond_angle_sd}\n"
            f"  dihedral_angle:{dihed_angle}"
        )
        
        # Try primary geometric rebuilding
        try:
            new_coor = self.calc_coordinates(
                dihedral_atom_coor.flatten(),
                bond_angle_coor.flatten(),
                ref_coor.flatten(),
                bond_length,
                bond_length_sd,
                np.deg2rad(bond_angle),
                np.deg2rad(bond_angle_sd),
                np.deg2rad(dihed_angle),
            )
            new_coor = [round(x, 3) for x in new_coor]
            logger.info(f"Rebuilt {atom} at {new_coor}")
            self.add_atom(atom, atom[0], new_coor)
            return
        except RuntimeError as e:
            logger.warning(f"Primary geometric rebuild failed for {atom}: {e}")
        
        # Try alternative bond angle atoms if available
        alternative_angles = []
        for angle in self._rotamers["bond_angle"]:
            if angle[0][1] == ref_atom and angle[0][2] == atom:
                if angle[0][0][0] != "H" and angle[0][0] != bond_angle_atom:
                    alternative_angles.append(angle)
        
        for alt_angle in alternative_angles:
            try:
                alt_bond_angle_atom = alt_angle[0][0]
                alt_bond_angle, alt_bond_angle_sd = alt_angle[1]
                
                if alt_bond_angle_atom not in self.name:
                    continue  # Skip if we don't have this atom
                
                alt_bond_angle_coor = self.coor[
                    np.argwhere(self.name == alt_bond_angle_atom)[0]
                ]
                
                logger.debug(f"Trying alternative bond angle atom {alt_bond_angle_atom} for {atom}")
                
                new_coor = self.calc_coordinates(
                    dihedral_atom_coor.flatten(),
                    alt_bond_angle_coor.flatten(),
                    ref_coor.flatten(),
                    bond_length,
                    bond_length_sd,
                    np.deg2rad(alt_bond_angle),
                    np.deg2rad(alt_bond_angle_sd),
                    np.deg2rad(dihed_angle),
                )
                new_coor = [round(x, 3) for x in new_coor]
                logger.info(f"Rebuilt {atom} at {new_coor} using alternative angle atom {alt_bond_angle_atom}")
                self.add_atom(atom, atom[0], new_coor)
                return
            except RuntimeError:
                continue  # Try next alternative
        
        # Try approximate positioning as last resort
        try:
            new_coor = self._approximate_atom_position(ref_atom, ref_coor, bond_length)
            logger.warning(f"Used approximate positioning for {atom} at {new_coor}")
            self.add_atom(atom, atom[0], new_coor)
            return
        except Exception:
            pass
        
        # If all attempts fail, log detailed warning and skip atom
        logger.warning(
            f"Unable to rebuild atom {atom} in {self}. "
            f"Tried primary geometric rebuild, {len(alternative_angles)} alternative bond angles, "
            f"and approximate positioning. Atom will be skipped. "
            f"Residue: {self.resn[0]} {self.resi[0]}, "
            f"Reference atom: {ref_atom}, Bond length: {bond_length}±{bond_length_sd}, "
            f"Bond angle: {bond_angle}±{bond_angle_sd}, Dihedral: {dihed_angle}"
        )

    def _approximate_atom_position(self, ref_atom, ref_coor, bond_length):
        """Approximate atom positioning as fallback when geometric rebuild fails.
        
        Uses simple vector math to place atom at approximate position based on
        reference atom and typical bond geometry.
        
        Args:
            ref_atom: Name of reference atom
            ref_coor: Coordinates of reference atom
            bond_length: Expected bond length
            
        Returns:
            np.ndarray: Approximate coordinates for the atom
        """
        # Find other atoms in the residue to determine general direction
        available_atoms = []
        for other_atom in self.name:
            if other_atom != ref_atom and other_atom not in ["H", "D"]:  # Skip hydrogens/deuterium
                other_idx = np.argwhere(self.name == other_atom)[0][0]
                other_coor = self.coor[other_idx]
                if not np.any(np.isnan(other_coor)):
                    available_atoms.append((other_atom, other_coor))
        
        if len(available_atoms) == 0:
            raise RuntimeError("No valid atoms available for approximate positioning")
        
        # Calculate center of mass of available atoms
        center_of_mass = np.mean([coor for _, coor in available_atoms], axis=0)
        
        # Direction vector from center of mass to reference atom
        com_to_ref = ref_coor.flatten() - center_of_mass
        com_to_ref_norm = np.linalg.norm(com_to_ref)
        
        if com_to_ref_norm > 0:
            # Place atom in direction away from center of mass
            direction = com_to_ref / com_to_ref_norm
        else:
            # If reference atom is at center of mass, use random direction
            direction = np.array([1.0, 0.0, 0.0])
        
        # Add some randomness to avoid perfect alignment
        random_perturbation = np.random.normal(0, 0.1, 3)
        direction += random_perturbation
        direction = direction / np.linalg.norm(direction)
        
        # Place atom at bond length distance in calculated direction
        approx_coor = ref_coor.flatten() + direction * bond_length
        
        return [round(x, 3) for x in approx_coor]

    @staticmethod
    def calc_coordinates(i, j, k, L, sig_L, theta, sig_theta, chi):
        """Calculate coords of an atom from three atomic positions and bond parms.

        Will permit deviations in bond length or theta to position an atom.
        Uses progressive expansion of search ranges for better success rate.

        Args:
            i (np.ndarray[float, shape=(3,)]): coords of atom 3-bonds away
            j (np.ndarray[float, shape=(3,)]): coords of atom 2-bonds away
            k (np.ndarray[float, shape=(3,)]): coords of neighbouring atom
            L (float): bond length
            sig_L (float): standard deviation of bond length
            theta (float): bond angle (in radians)
            sig_theta (float): standard deviation of bond angle (in radians)
            chi (float): dihedral angle (in radians)

        Returns:
            np.ndarray[float, shape=(3,): coords of atom
        """
        # Progressive expansion of search ranges for better success rate
        expansion_factors = [1.0, 1.5, 2.0, 3.0]
        search_points = [5, 8, 10, 12]  # Increase resolution as we expand
        
        for expansion_factor, n_points in zip(expansion_factors, search_points):
            # Expand search ranges
            expanded_sig_L = sig_L * expansion_factor
            expanded_sig_theta = sig_theta * expansion_factor
            
            # Generate parameter options with higher resolution
            theta_options_larger = np.linspace(theta, theta + expanded_sig_theta, n_points, endpoint=False)
            theta_options_smaller = np.linspace(
                theta, theta - expanded_sig_theta, n_points, endpoint=False
            )[1:]
            theta_options = [theta, *theta_options_larger, *theta_options_smaller]

            L_options_larger = np.linspace(L, L + expanded_sig_L, n_points, endpoint=False)
            L_options_smaller = np.linspace(L, L - expanded_sig_L, n_points, endpoint=False)[1:]
            L_options = [L, *L_options_larger, *L_options_smaller]

            tries = itl.product(theta_options, L_options)

            # Loop over parameters, and return the first success.
            for try_theta, try_L in tries:
                try:
                    coordinates = _RotamerResidue.position_from_bond_parms(
                        i, j, k, try_L, try_theta, chi
                    )
                except ValueError:
                    # If these parameters didn't work, try the next set.
                    continue
                else:
                    if expansion_factor > 1.0:
                        logger.debug(
                            f"Successfully rebuilt atom with expanded parameters: "
                            f"expansion_factor={expansion_factor}, L={try_L:.3f}, theta={try_theta:.3f}"
                        )
                    return coordinates

        # If we get here, we can't rebuild the atom according to our parameters.
        max_expansion = expansion_factors[-1]
        raise RuntimeError(
            f"Could not determine position. "
            f"Exhausted L ∈ [{L - sig_L * max_expansion:.2f}, {L + sig_L * max_expansion:.2f}] and "
            f"theta ∈ [{theta - sig_theta * max_expansion:.2f}, {theta + sig_theta * max_expansion:.2f}]"
        )

    @staticmethod
    def position_from_bond_parms(i, j, k, L, theta, chi):
        """Calculate coords of a 4th atom from 3 atomic coords and bond parms.

        Args:
            i (np.ndarray[float, shape=(3,)]): coords of atom 3-bonds away
            j (np.ndarray[float, shape=(3,)]): coords of atom 2-bonds away
            k (np.ndarray[float, shape=(3,)]): coords of neighbouring atom
            L (float): bond length
            theta (float): bond angle (in radians)
            chi (float): dihedral angle (in radians)

        Returns:
            np.ndarray[float, shape=(3,): coords of atom
        """

        # First, calculate distance vectors u = vec(ji); v = vec(kj)
        u = i - j
        v = j - k

        # Our task here is to find x, the distance vector < a_x, b_x, c_x >
        #     from k to our new atom.

        # Equation 1: plane (derived from θ)
        # ==================================
        #   x . v
        # ---------   = cosθ
        #  ‖x‖ ‖v‖
        #       x . v = ‖v‖ L cosθ
        #
        #     Let  C1 = ‖v‖ L cosθ
        #
        #       x . v = C1

        C1 = np.linalg.norm(v) * L * np.cos(theta)

        # If v is normalized
        norm_v = v / np.linalg.norm(v)
        norm_C1 = L * np.cos(theta)

        # Equation 2: plane (derived from χ)
        # ==================================
        # https://en.wikipedia.org/wiki/Dihedral_angle#Mathematical_background
        #        v . (( x × v ) × ( u × v ))
        #       ----------------------------- = sinχ
        #         ‖ v ‖ ‖ u × v ‖ ‖ x × v ‖
        #
        #     But, ‖ x × v ‖ = ‖x‖ ‖v‖ sinθ = ‖v‖ L sinθ
        #         v . (( x × v ) × ( u × v )) = ‖ u × v ‖ ‖v‖^2 L sinθ sinχ
        #     Flipping the cross products
        #         v . (( v × x ) × ( v × u )) = ‖ u × v ‖ ‖v‖^2 L sinθ sinχ
        #     Factoring a scalar triple product out of the vector product
        #              v . (( v . ( x × u ))v = ‖ u × v ‖ ‖v‖^2 L sinθ sinχ
        #     Noting that this is simply ( v . f v ), which is f * ‖v‖^2
        #           (( v . ( x × u )) * ‖v‖^2 = ‖ u × v ‖ ‖v‖^2 L sinθ sinχ
        #     Rotating the scalar triple product, cancelling ‖v‖^2
        #                       x . ( u × v ) = ‖ u × v ‖ L sinθ sinχ

        #     Let                           w = u × v
        #                                  C2 = ‖ u × v ‖ L sinθ sinχ
        #
        #                               x . w = C2

        w = np.cross(u, v)
        C2 = np.linalg.norm(np.cross(u, v)) * L * np.sin(theta) * np.sin(chi)

        # Normalizing this vector
        norm_w = w / np.linalg.norm(w)
        norm_C2 = L * np.sin(theta) * np.sin(chi)

        # Intersection of planes to construct a line
        # ==========================================
        # https://en.wikipedia.org/wiki/Plane_(geometry)#Line_of_intersection_between_two_planes

        # The cross product of the normals to the two planes will be a line
        #     direction, co-linear to the line of intersection.

        # The line of intersection between two planes
        #     Π_1: n1 . r = h1
        #     Π_2: n2 . r = h2
        #     where n_i are normalized can be written as:
        #
        #     r = k(n1 × n2) + r0
        #
        #     If n1 and n2 are orthonormal, r0 is
        #           r0 = n1 h1 + n2 h2

        # Since w = u × v, v ⟂ w, and
        r0 = norm_v * norm_C1 + norm_w * norm_C2

        # Calculate a unit vector along the line from v × w
        unit_vw = np.cross(v, w) / np.linalg.norm(np.cross(v, w))

        # Equation 3: sphere (derived from L)
        # ==================================
        # a_x^2 + b_x^2 + c_x^2 - L^2 = 0

        # Intersection of sphere with line to determine points
        # ====================================================
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

        # Here, we have our sphere:
        #     ‖x‖^2 = L^2
        # where x are the points on the sphere.

        # We also have a line:
        #     x = k l + r0
        # where x are points on the line,
        #       k is a distance along the line
        #       l is a unit vector along the line
        #       r0 is the origin of the line.

        # Combining these two equations
        #     L^2 = ‖ r0 + k l ‖^2
        #         = (r0 + k l) . (r0 + k l)
        #       0 = k^2 (l . l)  +  k (2 r0 . l)  +  (r0 . r0 - L^2)
        # which is a quadratic in k.
        # l is a unit vector, so:
        #       0 = k^2 (1)  +  k (2 r0 . l)  +  (r0 . r0 - L^2)

        b = 2 * np.dot(unit_vw, r0)
        c = np.dot(r0, r0) - L**2

        discriminant = b**2 - 4 * c
        if discriminant < 0:
            logger.debug(
                f"Bond parameters too restrictive!\n"
                f"  u, v: {u, v}\n"
                f"  r0: {r0}\n"
                f"  unit_vw: {unit_vw}\n"
                f"  b, c: {b, c}\n"
                f"  discriminant: {discriminant:.2e}"
            )
            raise ValueError(
                f"Could not determine position to rebuild atom. "
                f"Discriminant: {discriminant:.2e}"
            )

        # Solve the quadratic, and determine 2 potential atom positions.
        # Resolve ambiguity of chi with 2-arg arctan.
        # Translate correct coordinate (it was calc'd relative to k).
        positions = r0 + unit_vw * np.roots([1.0, b, c])[:, np.newaxis]
        calc_chis = [
            np.arctan2(
                np.dot(v, np.cross(np.cross(pos, v), np.cross(u, v))),
                np.linalg.norm(v) * np.dot(np.cross(u, v), np.cross(pos, v)),
            )
            for pos in positions
        ]
        correct_chi = np.isclose(calc_chis, chi)

        # We have calculated atom positions. If there are no correct_chi, this
        # seems to occur from rounding errors at the edges of (-π, π].
        if correct_chi.sum() == 0:
            logger.debug(
                f"No valid chi results:\n"
                f"  chi / calc_chis: {chi}/({calc_chis})\n"
                f"  correct_chi: {correct_chi}"
            )
            raise ValueError(f"Couldn't determine a matching chi.")

        x = positions[correct_chi][
            0
        ]  # if discriminant~0, this has identical sols. Take the first.
        x += k
        return x

    def add_atom(self, name, element, coor):
        """Add or update an atom in the residue.

        Args:
            name: The name of the atom
            element: The element type
            coor: The coordinates of the atom

        If the atom already exists, its coordinates will be updated.
        If the atom doesn't exist, it will be added to the end of the residue.
        If the atom exists but has NaN coordinates, those will be properly updated.
        """
        existing_atom_indices = np.argwhere(self.name == name)
        if existing_atom_indices.size > 0:
            idx = self._selection[existing_atom_indices[0][0]]
            # Check if existing coordinates are NaN - if so, log this specific case
            if np.any(np.isnan(self._coor[idx])):
                logger.info(
                    f"Updating {name} with NaN coordinates to valid coordinates: {coor}"
                )

            # Update coordinates and active status
            coor_array = getattr(self, "_coor").copy()
            coor_array[idx] = np.array(coor)
            active_array = getattr(self, "_active").copy()
            active_array[idx] = True
            setattr(self, "_coor", coor_array)
            setattr(self, "_active", active_array)

            # Calculate and update B-factor
            current_b_factors = getattr(self, "_b")
            valid_b_factors = current_b_factors[~np.isnan(current_b_factors)]
            mean_b = np.mean(valid_b_factors) if valid_b_factors.size > 0 else 40.0
            b_array = getattr(self, "_b").copy()
            b_array[idx] = mean_b
            setattr(self, "_b", b_array)

            # Calculate and update Occupancy
            current_q = getattr(self, "_q")
            valid_q = current_q[~(current_q == 0)]
            if valid_q.size > 0 and np.all(valid_q == 1.0):
                target_q = 1.0
            elif valid_q.size > 0:
                target_q = np.mean(valid_q)
            else:
                target_q = 1.0
            q_array = getattr(self, "_q").copy()
            q_array[idx] = target_q
            setattr(self, "_q", q_array)

            # Update coordinates, active, b, and q in all parent levels
            current = self
            while current.parent is not None:
                current.parent.data["coor"][idx] = np.array(coor)
                current.parent.data["active"][idx] = True
                current.parent.data["b"][idx] = mean_b
                current.parent.data["q"][idx] = target_q
                current = current.parent

            logger.info(
                f"Updated {name} coordinates to {coor}, B-factor to {mean_b:.2f}, Occupancy to {target_q:.2f}"
            )
            return

        # Adding a new atom
        index = self._selection[-1]
        if index < len(self.data["record"]):
            index = len(self.data["record"]) - 1

        # Pre-calculate mean B-factor and target Q for the new atom
        current_b_factors = getattr(self, "_b")
        valid_b_factors = current_b_factors[~np.isnan(current_b_factors)]
        mean_b_new = np.mean(valid_b_factors) if valid_b_factors.size > 0 else 40.0

        current_q = getattr(self, "_q")
        valid_q = current_q[~np.isnan(current_q)]
        if valid_q.size > 0 and np.all(valid_q == 1.0):
            target_q_new = 1.0
        elif valid_q.size > 0:
            target_q_new = np.mean(valid_q)
        else:
            target_q_new = 1.0

        for attr in self.data:
            if attr == "e":
                setattr(self, "_" + attr, np.append(getattr(self, "_" + attr), element))
            elif attr == "atomid":
                setattr(
                    self, "_" + attr, np.append(getattr(self, "_" + attr), index + 1)
                )
            elif attr == "name":
                setattr(self, "_" + attr, np.append(getattr(self, "_" + attr), name))
            elif attr == "coor":
                setattr(
                    self,
                    "_" + attr,
                    np.append(
                        getattr(self, "_" + attr), np.expand_dims(coor, axis=0), axis=0
                    ),
                )
            elif attr == "b":
                setattr(
                    self, "_" + attr, np.append(getattr(self, "_" + attr), mean_b_new)
                )
            elif attr == "q":
                setattr(
                    self, "_" + attr, np.append(getattr(self, "_" + attr), target_q_new)
                )
            elif attr == "active":  # Ensure new atoms are active
                setattr(self, "_" + attr, np.append(getattr(self, "_" + attr), True))
            else:  # Append the last value for other attributes (resn, chain, etc.)
                setattr(
                    self,
                    "_" + attr,
                    np.append(getattr(self, "_" + attr), getattr(self, attr)[-1]),
                )
        selection = np.append(
            self.__dict__["_selection"].astype(int), int(index + 1)
        )  # ensuring this is not a float
        setattr(self, "_selection", selection)
        setattr(self, "natoms", self.natoms + 1)

    def reorder(self):
        for idx, atom2 in enumerate(
            self._rotamers["atoms"] + self._rotamers["hydrogens"]
        ):
            if self.name[idx] != atom2:
                idx2 = np.argwhere(self.name == atom2)[0]
                index = np.ndarray((1,), dtype="int")
                index[0,] = np.array(self.__dict__["_selection"][0])
                for attr in [
                    "record",
                    "name",
                    "b",
                    "q",
                    "coor",
                    "resn",
                    "resi",
                    "icode",
                    "e",
                    "charge",
                    "chain",
                    "altloc",
                ]:
                    (
                        self.__dict__["_" + attr][index + idx],
                        self.__dict__["_" + attr][index + idx2],
                    ) = (
                        self.__dict__["_" + attr][index + idx2],
                        self.__dict__["_" + attr][index + idx],
                    )

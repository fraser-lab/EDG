import numpy as np
import copy
import itertools
import os

from .modules.base_structure import _BaseStructure, PDBFile
from .modules.ligand import _Ligand
from .modules.residue import _Residue, _RotamerResidue, residue_type
from .modules.rotamers import ROTAMERS
from .modules.math import Rz
from ..utils.normalize_to_precision import normalize_to_precision
from .modules.mmciffile import mmCIFFile

class Structure(_BaseStructure):
    """Class with access to underlying PDB hierarchy."""

    def __init__(self, data, **kwargs):
        for attr in self.REQUIRED_ATTRIBUTES:
            if attr not in data:
                raise ValueError(
                    f"Not all attributes are given to " f"build the structure: {attr}"
                )
        super().__init__(data, **kwargs)
        self._chains = []
        self.resolution = kwargs.get("resolution", None)

    @classmethod
    def fromfile(cls, fname, use_auth=False):
        extension = os.path.splitext(fname)[1].lower()

        if extension == ".cif":
            if isinstance(fname, mmCIFFile):
                pdbfile = fname
            else:
                pdbfile = mmCIFFile.read(fname, use_auth=use_auth)
        elif extension == ".pdb":
            if isinstance(fname, PDBFile):
                pdbfile = fname
            else:
                pdbfile = PDBFile.read(fname)
        else:
            raise ValueError(
                f"fname extension is not valid: {extension} must be one of .cif, .pdb"
            )
        dd = pdbfile.coor
        n_active_atoms = len(dd["x"])
        if pdbfile.missing_atoms:
            for key in pdbfile.missing_atoms:
                # TODO: check if this handles clash analysis
                dd[key].extend(pdbfile.missing_atoms[key])
        data = {}
        for attr, array in dd.items():
            if attr in "xyz":
                continue
            data[attr] = np.asarray(array)
        coor = np.asarray(list(zip(dd["x"], dd["y"], dd["z"])), dtype=np.float64)

        dl = pdbfile.link
        link_data = {}
        for attr, array in dl.items():
            link_data[attr] = np.asarray(array)

        data["coor"] = coor
        # Add an active array, to check for collisions and density creation.
        data["active"] = np.ones(len(dd["x"]), dtype=bool)
        data["active"][n_active_atoms:] = False  # Set the missing atoms to inactive
        if pdbfile.anisou:
            natoms = len(data["record"])
            anisou = np.zeros((natoms, 6), float)
            anisou_atomid = pdbfile.anisou["atomid"]
            n = 0
            us = ["u00", "u11", "u22", "u01", "u02", "u12"]
            nanisou = len(anisou_atomid)
            for i, atomid in enumerate(pdbfile.coor["atomid"]):
                if n < nanisou and atomid == anisou_atomid[n]:
                    anisou[i] = [pdbfile.anisou[u][n] for u in us]
                    n += 1
            for n, key in enumerate(us):
                data[key] = anisou[:, n]

        if pdbfile.cryst1:
            from ..qfit.unitcell import UnitCell

            c = pdbfile.cryst1
            # FIXME Create correct SpaceGroup if not automatically found.
            values = [c[x] for x in ["a", "b", "c", "alpha", "beta", "gamma", "spg"]]
            cls.unit_cell = UnitCell(*values)

        return cls(
            data,
            link_data=link_data,
            scale=pdbfile.scale,
            cryst_info=pdbfile.cryst_info,
            resolution=pdbfile.resolution,
        )

    @classmethod
    def fromstructurelike(cls, structure_like):
        data = {}
        for attr in structure_like.data:
            data[attr] = getattr(structure_like, attr)
        return cls(data)

    @classmethod
    def empty(self, cls):
        data = {}
        for attr in self.REQUIRED_ATTRIBUTES:
            data[attr] = []
        return cls(data)

    def __getitem__(self, key):
        if not self._chains:
            self.build_hierarchy()
        if isinstance(key, int):
            nchains = len(self._chains)
            if key < 0:
                key = key + nchains
            if key >= nchains or key < 0:
                raise IndexError("Selection out of range.")
            else:
                return self._chains[key]
        elif isinstance(key, str):
            for chain in self._chains:
                if key == chain.id:
                    return chain
            raise KeyError
        else:
            raise TypeError

    def __repr__(self):
        if not self._chains:
            self.build_hierarchy()
        return f"Structure: {self.natoms} atoms"

    @property
    def atoms(self):
        indices = self._selection
        if indices is None:
            indices = range(self.natoms)
        for index in indices:
            yield _Atom(self.data, index)

    @property
    def chains(self):
        if not self._chains:
            self.build_hierarchy()
        return self._chains

    @property
    def residue_groups(self):
        for chain in self.chains:
            for rg in chain.residue_groups:
                yield rg

    @property
    def residues(self):
        for chain in self.chains:
            for conformer in chain.conformers:
                for residue in conformer.residues:
                    yield residue

    @property
    def single_conformer_residues(self):
        Dict = {}
        for chain in self.chains:
            if chain.chain[0] not in Dict:
                Dict[chain.chain[0]] = {}
            for conformer in chain.conformers:
                for residue in conformer.residues:
                    if residue.resi[0] not in Dict[chain.chain[0]]:
                        yield residue
                        Dict[chain.chain[0]][residue.resi[0]] = 1

    @property
    def segments(self):
        for chain in self.chains:
            for conformer in chain.conformers:
                for segment in conformer.segments:
                    yield segment

    def build_hierarchy(self):
        # Build up hierarchy starting from chains
        chainids = np.unique(self.chain).tolist()
        self._chains = []
        for chainid in chainids:
            selection = self.select("chain", chainid)
            chain = _Chain(self.data, selection=selection, parent=self, chainid=chainid)
            self._chains.append(chain)

    def combine(self, structure):
        """Combines two structures into one"""
        data = {}
        for attr in self.data:
            array1 = getattr(self, attr)
            array2 = getattr(structure, attr)
            combined = np.concatenate((array1, array2))
            data[attr] = combined
        return Structure(data)

    def collapse_backbone(self, resid, chainid):
        """Collapses the backbone atoms of a given residue"""
        data = {}
        # determine altloc to keep
        sel_str = f"resi {resid} and chain {chainid}"
        conformers = [self.extract(sel_str)]
        altlocs = []
        for i in range(0, len(conformers)):
            altlocs = np.append(altlocs, conformers[i].altloc[0])
        altloc_keep = np.unique(altlocs)[0]
        altlocs_remove = np.unique(altlocs)[1:]
        mask = (
            (self.data["resi"] == resid)
            & (self.data["chain"] == chainid)
            & np.isin(self.data["name"], ["CA", "C", "N", "O", "H", "HA"])
            & np.isin(self.data["altloc"], altlocs_remove)
        )
        mask2 = (
            (self.data["resi"] == resid)
            & (self.data["chain"] == chainid)
            & np.isin(self.data["name"], ["CA", "C", "N", "O"])
            & np.isin(self.data["altloc"], altloc_keep)
        )

        for attr in self.data:
            array1 = getattr(self, attr)
            if attr == "altloc":
                array1[mask2] = ""
            if attr == "q":
                array1[mask2] = 1.0
            data[attr] = array1[~mask]

        return Structure(data)

    def get_backbone(self):
        """Return the backbone atoms of a given residue"""
        data = {}
        mask = (self.data["resi"] == self.resi[0]) & np.isin(
            self.data["name"], ["CA", "C", "N", "O", "H", "HA"], invert=True
        )
        for attr in self.data:
            array1 = getattr(self, attr)
            data[attr] = array1[~mask]
        return Structure(data)

    def set_backbone_occ(self):
        """Set the "backbone" occupancy to 0 and the occupancy of
        other atoms to 1.0"""
        data = {}
        if self.resn[0] == "GLY":
            # "Backbone" atoms for the residue:
            mask = np.isin(self.data["name"], ["CA", "C", "N", "H", "HA"])
            # Non-"backbone" atoms for the residue:
            mask2 = np.isin(self.data["name"], ["CA", "C", "N", "H", "HA"], invert=True)
        else:
            # "Backbone" atoms for the residue:
            mask = np.isin(self.data["name"], ["CA", "C", "N", "O", "H", "HA"])
            # Non-"backbone" atoms for the residue:
            mask2 = np.isin(
                self.data["name"], ["CA", "C", "N", "O", "H", "HA"], invert=True
            )
        for attr in self.data:
            array1 = getattr(self, attr)
            if attr == "q":
                array1[mask] = 0.0
                array1[mask2] = 1.0
            data[attr] = array1
        return Structure(data)

    def register(self, attr, array):
        """Register array attribute"""
        if self.parent is not None:
            msg = "This structure has a parent, registering a new array \
                    is not allowed."
            raise ValueError(msg)

        self.data[attr] = array
        hattr = "_" + attr
        setattr(self, hattr, array)
        setattr(self.__class__, attr, self._structure_property(hattr))

    def reorder(self):
        ordering = []
        for chain in self.chains:
            for rg in chain.residue_groups:
                if rg.resn[0] in ROTAMERS:
                    # There are some perverted cases apparently in the PDB that
                    # have more than 1 resname in a specific residue 'slot'
                    if len(set(rg.resn)) > 1:
                        resi = rg.resi[0]
                        icode = rg.icode[0]
                        raise RuntimeError(
                            f"Residue has more than 1 name. \
                                             {resi}{icode}"
                        )
                    rotamer = ROTAMERS[rg.resn[0]]
                    atom_order = rotamer["atoms"] + rotamer["hydrogens"]
                    atomnames = list(rg.name)
                    # Check if all atomnames are standard. We don't touch the
                    # ordering if it isnt recognized.
                    for atom in atomnames:
                        if atom not in atom_order:
                            break
                    # Check if the residue has alternate conformers. If the
                    # number of altlocs is equal to the whole residue, sort
                    # it after one another. If its just a few, sort it like a
                    # zipper.
                    altlocs = sorted(list(set(rg.altloc)))
                    try:
                        altlocs.remove("")
                    except ValueError:
                        pass
                    naltlocs = len(altlocs)
                    if naltlocs < 2:
                        residue_ordering = []
                        for atom in atom_order:
                            try:
                                index = atomnames.index(atom)
                                residue_ordering.append(index)
                            except ValueError:
                                continue
                        residue_ordering = rg._selection[residue_ordering]
                    else:
                        atoms_per_altloc = []
                        zip_atoms = True
                        if rg.select("altloc", "").size == 0:
                            for altloc in altlocs:
                                nsel = rg.select("altloc", altloc).size
                                atoms_per_altloc.append(nsel)
                            zip_atoms = not all(
                                a == atoms_per_altloc[0] for a in atoms_per_altloc
                            )
                        residue_orderings = []
                        for altloc in altlocs:
                            altconf = rg.extract("altloc", ("", altloc))
                            residue_ordering = []
                            atomnames = list(altconf.name)
                            for atom in atom_order:
                                try:
                                    index = atomnames.index(atom)
                                    residue_ordering.append(index)
                                except ValueError:
                                    continue
                            residue_ordering = altconf._selection[residue_ordering]
                            residue_orderings.append(residue_ordering)
                        if (
                            zip_atoms
                            and len(list(set([len(x) for x in residue_orderings]))) == 1
                        ):
                            residue_ordering = list(zip(*residue_orderings))
                            residue_ordering = np.concatenate(residue_ordering)
                        else:
                            residue_ordering = np.concatenate(residue_orderings)
                        # Now remove duplicates while keeping order
                        seen = set()
                        residue_ordering = [
                            x
                            for x in residue_ordering
                            if not (x in seen or seen.add(x))
                        ]
                    ordering.append(residue_ordering)
                    continue
                for ag in rg.atom_groups:
                    ordering.append(ag._selection)
        ordering = np.concatenate(ordering)
        data = {}
        for attr, value in self.data.items():
            data[attr] = value[ordering]
        return Structure(
            data, link_data=self.link_data, scale=self.scale, cryst_info=self.cryst_info
        )

    def normalize_occupancy(self):
        """This function will scale the occupancy of protein residues to make the sum(occ) equal to 1 for all.
        The goal of this function is to determine if the sum(occ) of each residue coming out of qFit residue or segement is < 1 (as it can be due to QP),
        and scale each alt conf occupancy for the residue to sum to one.
        To accomplish this, if the sum(occ) is less than 1, then we will divide each conformer occupancy by the sum(occ).
        """
        multiconformer = copy.deepcopy(self)
        data = self.data
        for chain in multiconformer:
            for residue in chain:
                altlocs = list(set(residue.altloc))
                mask = None
                if len(altlocs) == 1:  # confirm occupancy = 1
                    mask = (self.data["resi"] == residue.resi[0]) & (
                        self.data["chain"] == residue.chain[0]
                    )
                    for attr in data:
                        array = getattr(multiconformer, attr)
                        if attr == "q":
                            array[mask] = 1.0
                        elif attr == "altloc":
                            array[mask] = ""
                        data[attr] = array
                    multiconformer = Structure(data)
                else:
                    new_occ = []
                    if "" in altlocs and len(altlocs) > 1:
                        mask = (
                            (self.data["resi"] == residue.resi[0])
                            & (self.data["chain"] == residue.chain[0])
                            & (self.data["altloc"] == "")
                        )
                        for attr in data:
                            array = getattr(multiconformer, attr)
                            if attr == "q":
                                array[mask] = 1.0
                            elif attr == "altloc":
                                array[mask] = ""
                            data[attr] = array
                        multiconformer = Structure(data)
                        altlocs.remove("")
                    sel_str = f"resi {residue.resi[0]} and chain {residue.chain[0]} and altloc "
                    conformers = [self.extract(sel_str + x) for x in altlocs]
                    alt_sum = 0
                    for i in range(0, len(conformers)):
                        alt_sum += np.round(conformers[i].q[0], 2)
                        new_occ = np.append(new_occ, (np.round(conformers[i].q[0], 2)))
                    if alt_sum != 1.0:  # we need to normalize
                        new_occ = []
                        for i in range(0, len(altlocs)):
                            new_occ = np.append(
                                new_occ, ((np.round(conformers[i].q[0], 2)) / alt_sum)
                            )
                        new_occ = normalize_to_precision(
                            np.array(new_occ), 2
                        )  # deal with imprecision
                    for i in range(0, len(new_occ)):
                        mask = (
                            (self.data["resi"] == residue.resi[0])
                            & (self.data["chain"] == residue.chain[0])
                            & (self.data["altloc"] == altlocs[i])
                        )
                        for attr in data:
                            array = getattr(multiconformer, attr)
                            if attr == "q":
                                array[mask] = new_occ[i]
                            data[attr] = array
                        multiconformer = Structure(data)
        return Structure(data)

    def remove_conformer(self, resi, chain, altloc1, altloc2):
        data = {}
        mask = (
            (self.data["resi"] == resi)
            & (self.data["chain"] == chain)
            & (self.data["altloc"] == altloc1)
        )
        mask2 = (
            (self.data["resi"] == resi)
            & (self.data["chain"] == chain)
            & (self.data["altloc"] == altloc2)
        )

        for attr in self.data:
            array1 = getattr(self, attr)
            if attr == "q":
                array1[mask] = array1[mask] + array1[mask2]
            data[attr] = array1[~mask2]

        return Structure(data)

    def remove_identical_conformers(self, rmsd_cutoff=0.01):
        multiconformer = copy.deepcopy(self)
        for chain in multiconformer:
            for residue in chain:
                altlocs = list(set(residue.altloc))
                try:
                    altlocs.remove("")
                except ValueError:
                    pass
                sel_str = (
                    f"resi {residue.resi[0]} and chain {residue.chain[0]} and altloc "
                )
                conformers = [multiconformer.extract(sel_str + x) for x in altlocs]
                if len(set(altlocs)) == 1:
                    continue
                else:
                    removed_conformers = []  # list of all conformer that are removed
                    for conf_a, conf_b in itertools.combinations(conformers, 2):
                        if conf_a.altloc[0] in removed_conformers:
                            continue  # we have already removed this conformer
                        else:
                            rmsd = np.sqrt(np.mean((conf_a.coor - conf_b.coor) ** 2))
                            if rmsd > rmsd_cutoff:
                                continue
                            else:
                                multiconformer = multiconformer.remove_conformer(
                                    residue.resi[0],
                                    residue.chain[0],
                                    conf_a.altloc[0],
                                    conf_b.altloc[0],
                                )
                                removed_conformers.append(conf_b.altloc[0])
        return multiconformer

    def remove_alternative_conformations(self):
        """Remove alternative conformations from the structure"""
        structure = copy.deepcopy(self)
        for chain in structure:
            for residue in chain:
                altlocs = sorted(list(set(residue.altloc)))
                resi = residue.resi[0]
                chainid = residue.chain[0]
                if len(altlocs) > 1:
                    try:
                        altlocs.remove(
                            ""
                        )  # FIXME: I think this will remove "missing atoms" that I care about
                    except ValueError:
                        pass
                    for altloc in altlocs[1:]:
                        sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
                        sel_str = f"not ({sel_str})"
                        structure = structure.extract(sel_str)

        non_zero_occ_idx = np.where(structure.data['q'] > 0)[0]
        structure.data['q'][non_zero_occ_idx] = 1.0
        structure.data['altloc'][:] = ""

        return structure

    def remove_hydrogens(self):
        """Remove hydrogen atoms from the structure.

        Returns
        -------
        Structure
            New structure with hydrogen atoms removed
        """
        selection = self.select("e", "H", "!=")
        return self.extract(selection)

    def keep_protein(self):
        """Keep only protein atoms.

        Returns
        -------
        Structure
            New structure containing only protein atoms
        """
        protein_resn = [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "MSE",
            "CSD",
            "CSO",
            "HYP",
            "SEC",
            "BMT",
            "AIB",
            "MLY",
            "SEP",
            "TPO",
            "PTR",
        ]
        selection = self.select("resn", protein_resn)
        return self.extract(selection)

    def keep_polymer(self):
        """Keep only polymer chains (protein or nucleic acid).

        Returns
        -------
        Structure
            New structure containing only polymer chains
        """
        polymer_resn = [
            # Protein residues
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "MSE",
            "CSD",
            "CSO",
            "HYP",
            "SEC",
            "BMT",
            "AIB",
            "MLY",
            "SEP",
            "TPO",
            "PTR",
            # Nucleic acids
            "A",
            "C",
            "G",
            "U",
            "T",
            "DA",
            "DC",
            "DG",
            "DT",
            "DI",
            "PSU",
            "I",
            "5MC",
            "OMC",
        ]
        selection = self.select("resn", polymer_resn)
        return self.extract(selection)

    def remove_water(self):
        """Remove water molecules.

        Returns
        -------
        Structure
            New structure with water molecules removed
        """
        selection = self.select("resn", ["HOH", "WAT", "H2O"], "!=")
        return self.extract(selection)

    def remove_ligands(self):
        """Remove ligands (non-polymer molecules).

        Returns
        -------
        Structure
            New structure with ligands removed
        """
        selection = self.select(
            "record", "HETATM", "!="
        )  # NOTE: This will delete noncanonical amino acids
        return self.extract(selection)

    def clean_structure(
        self,
        keep_type="all",
        remove_h=True,
        remove_water_molecules=True,
        remove_all_ligands=True,
    ):
        """Clean structure by selectively removing atoms. Default removes water, hydrogen, and ligands.

        Parameters
        ----------
        keep_type : str
            What type of molecules to keep: 'protein', 'polymer', or 'all'
        remove_h : bool
            Remove hydrogen atoms
        remove_water_molecules : bool
            Remove water molecules
        remove_all_ligands : bool
            Remove all ligands (when keep_type is 'all') (will remove noncanonical AAs)

        Returns
        -------
        Structure
            Cleaned structure
        """
        structure = self.copy()

        if remove_h:
            structure = structure.remove_hydrogens()

        if keep_type == "protein":
            structure = structure.keep_protein()
        elif keep_type == "polymer":
            structure = structure.keep_polymer()
        elif keep_type == "all":
            if remove_water_molecules:
                structure = structure.remove_water()
            if remove_all_ligands:
                structure = structure.remove_ligands()

        return structure.reorder()

    def complete_residues(self):
        """Complete residues with missing atoms.

        Returns
        -------
        Structure
            Structure with missing atoms completed
        """
        structure = self.copy()
        if not np.all(structure.active):
            if not np.any(structure.active):
                print(f"No active atoms in {structure}")
                return structure
            else:
                for residue in structure.residues:
                    if not np.all(residue.active[:4]):
                        print(f"Skipping missing residue {residue.resn[0]} {residue.resi[0]} with missing backbone atoms")
                        continue
                    if not np.all(residue.active):
                        print(f"Completing residue {residue.resn[0]} {residue.resi[0]}")
                        residue.complete_residue()
        return structure

    @property
    def n_residue_conformers(self):
        """Total of conformers over all residues in the structure (exclude heteroatoms).

        Returns:
            int
        """
        total_altlocs = 0
        for rg in self.extract("record", "ATOM").residue_groups:
            altlocs = set(rg.altloc)
            if "" in altlocs and len(altlocs) > 1:
                altlocs.remove("")
            total_altlocs += len(altlocs)
        return total_altlocs

    @property
    def n_residues(self):
        """Number of residues in the structure (exclude heteroatoms).

        Required because Structure.residue_groups is a generator.

        Returns:
            int
        """
        residue_groups = self.extract("record", "ATOM").residue_groups
        return sum(1 for _ in residue_groups)

    def average_conformers(self):
        """Average number of conformers over the structure (exclude heteroatoms).

        Returns:
            float
        """
        return self.n_residue_conformers / self.n_residues

    def _init_clash_detection(self):
        # Setup the condensed distance based arrays for clash detection and
        # fill them
        self._ndistances = self.natoms * (self.natoms - 1) // 2
        self._clash_mask = np.ones(self._ndistances, bool)
        self._clash_radius2 = np.zeros(self._ndistances, float)
        radii = self.covalent_radius
        offset = self.natoms * (self.natoms - 1) // 2
        for i in range(self.natoms - 1):
            rotamers = ROTAMERS[self.resn[i]]
            bonds = rotamers["bonds"]
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
                if self.resi[i] == self.resi[j] and self.chain[i] == self.chain[j]:
                    if bond1 in bonds or bond2 in bonds:
                        self._clash_mask[index] = False
                elif (
                    self.resi[i] + 1 == self.resi[j] and self.chain[i] == self.chain[j]
                ):
                    if name1 == "C" and self.name[j] == "N":
                        self._clash_mask[index] = False
                elif self.e[i] == "S" and self.e[j] == "S":
                    self._clash_mask[index] = False
        self._clash_radius2 *= self._clash_radius2
        self._clashing = np.zeros(self._ndistances, bool)
        self._dist2_matrix = np.empty(self._ndistances, float)

        # All atoms are active from the start
        self.active = np.ones(self.natoms, bool)
        self._active_mask = np.ones(self._ndistances, bool)

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

    def extract_neighbors(self, residue, distance=5.0):
        # Remove the residue of interest from the structure:
        sel_str = f"resi {residue.resi[0]} and chain {residue.chain[0]}"
        sel_str = f"not ({sel_str})"

        # Identify all atoms within distance of the residue of interest
        neighbors = self.extract(sel_str)
        mask = neighbors.coor[:, 1] < -np.inf
        for coor in residue.coor:
            diffs = neighbors.coor - np.array(coor)
            dists = np.linalg.norm(diffs, axis=1)
            mask = np.logical_or(mask, dists < distance)

        # Copy the attributes of the masked atoms:
        data = {}
        for attr in neighbors.data:
            array1 = getattr(neighbors, attr)
            data[attr] = array1[mask]

        # Return the new object:
        return Structure(data)


class _Chain(_BaseStructure):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.id = kwargs["chainid"]
        self._residue_groups = []
        self._conformers = []
        self.conformer_ids = np.unique(self.altloc)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            if not self._residue_groups:
                self.build_hierarchy()
        nresidues = len(self._residue_groups)
        if isinstance(key, int):
            if key < 0:
                key += nresidues
            if key >= nresidues or key < 0:
                raise IndexError
            else:
                return self._residue_groups[key]
        elif isinstance(key, slice):
            start = key.start
            end = key.end
            if start < 0:
                start += nresidues
            if end < 0:
                end += nresidues
            return self._residue_groups[start:end]
        elif isinstance(key, str):
            if not self._conformers:
                self.build_conformers()
            for conformer in self._conformers:
                if conformer.id == key:
                    return conformer
            raise KeyError
        else:
            raise TypeError

    def __repr__(self):
        return "Chain: {chainid}".format(chainid=self.id)

    @property
    def conformers(self):
        if not self._conformers:
            self.build_conformers()
        return self._conformers

    @property
    def residue_groups(self):
        if not self._residue_groups:
            self.build_hierarchy()
        return self._residue_groups

    def build_hierarchy(self):
        resi = self.resi
        # order = np.argsort(resi)
        # resi = resi[order]
        # icode = self.icode[order]
        icode = self.icode
        # A residue group is a collection of entries that have a unique
        # chain, resi, and icode
        # Do all these order tricks to keep the resid ordering correct
        cadd = np.char.add
        residue_group_ids = cadd(cadd(resi.astype(str), "_"), icode)
        residue_group_ids, ind = np.unique(residue_group_ids, return_index=True)
        order = np.argsort(ind)
        residue_group_ids = residue_group_ids[order]
        self._residue_groups = []
        self._residue_group_ids = []
        for residue_group_id in residue_group_ids:
            resi, icode = residue_group_id.split("_")
            resi = int(resi)
            selection = self.select("resi", resi)
            selection = np.intersect1d(selection, self.select("icode", icode))
            residue_group = _ResidueGroup(
                self.data, selection=selection, parent=self, resi=resi, icode=icode
            )
            self._residue_groups.append(residue_group)
            self._residue_group_ids.append((resi, icode))

    def build_conformers(self):
        altlocs = np.unique(self.altloc)
        self._conformers = []
        if altlocs.size > 1 or altlocs[0] != "":
            main_selection = self.select("altloc", "")
            for altloc in altlocs:
                if not altloc:
                    continue
                altloc_selection = self.select("altloc", altloc)
                selection = np.union1d(main_selection, altloc_selection)
                conformer = _Conformer(
                    self.data, selection=selection, parent=self, altloc=altloc
                )
                self._conformers.append(conformer)
        else:
            conformer = _Conformer(
                self.data, selection=self._selection, parent=self, altloc=""
            )
            self._conformers.append(conformer)


class _ResidueGroup(_BaseStructure):
    """Guarantees a group with similar resi and icode."""

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.id = (kwargs["resi"], kwargs["icode"])
        self._atom_groups = []

    @property
    def atom_groups(self):
        if not self._atom_groups:
            self.build_hierarchy()
        return self._atom_groups

    def build_hierarchy(self):
        # An atom group is a collection of entries that have a unique
        # chain, resi, icode, resn and altloc
        cadd = np.char.add
        self.atom_group_ids = np.unique(cadd(cadd(self.resn, "_"), self.altloc))
        self._atom_groups = []
        for atom_group_id in self.atom_group_ids:
            resn, altloc = atom_group_id.split("_")
            selection = self.select("resn", resn)
            if altloc:
                altloc_selection = self.select("altloc", altloc)
                selection = np.intersect1d(
                    selection, altloc_selection, assume_unique=True
                )
            atom_group = _AtomGroup(
                self.data, selection=selection, parent=self, resn=resn, altloc=altloc
            )
            self._atom_groups.append(atom_group)

    def __repr__(self):
        resi, icode = self.id
        string = "ResidueGroup: resi {}".format(resi)
        if icode:
            string += ":{}".format(icode)
        return string


class _AtomGroup(_BaseStructure):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.id = (kwargs["resn"], kwargs["altloc"])

    def __repr__(self):
        string = "AtomGroup: {} {}".format(*self.id)
        return string


class _Atom:
    def __init__(self, data, index):
        self.index = index
        for attr, array in data.items():
            hattr = "_" + attr
            setattr(self, hattr, array)
            prop = self._atom_property(hattr)
            setattr(_Atom, attr, prop)

    def _atom_property(self, property_name, docstring=None):
        def getter(self):
            return self.__getattribute__(property_name)[self.index]

        def setter(self, value):
            getattr(self, property_name)[self.index] = value

        return property(getter, setter, doc=docstring)

    def __repr__(self):
        return f"Atom: {self.index}"


class _Conformer(_BaseStructure):
    """Guarantees a single consistent conformer."""

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.id = kwargs["altloc"]
        self._residues = []
        self._segments = []

    def __getitem__(self, arg):
        if not self._residues:
            self.build_residues()
        if isinstance(arg, int):
            key = (arg, "")
        elif isinstance(arg, tuple):
            key = arg
        for residue in self._residues:
            if residue.id == key:
                return residue
        raise ValueError("Residue id not found.")

    def __repr__(self):
        return "Conformer {}".format(self.id)

    @property
    def ligands(self):
        if not self._residues:
            self.build_residues()
        _ligands = [
            l for l in self._residues if isinstance(l, _Ligand) and l.natoms > 3
        ]
        return _ligands

    @property
    def residues(self):
        if not self._residues:
            self.build_residues()
        return self._residues

    @property
    def segments(self):
        if not self._segments:
            self.build_segments()
        return self._segments

    def build_residues(self):
        # A residue group is a collection of entries that have a unique
        # chain, resi, and icode
        # Do all these order tricks in order to keep the resid ordering correct
        resi = self.resi
        icode = self.icode
        cadd = np.char.add
        residue_ids = cadd(cadd(resi.astype(str), "_"), icode)
        residue_ids, ind = np.unique(residue_ids, return_index=True)
        order = np.argsort(ind)
        residue_ids = residue_ids[order]
        self._residues = []
        for residue_id in residue_ids:
            resi, icode = residue_id.split("_")
            resi = int(resi)
            selection = self.select("resi", resi)
            icode_selection = self.select("icode", icode)
            selection = np.intersect1d(selection, icode_selection, assume_unique=True)
            residue = self.extract(selection)
            rtype = residue_type(residue)
            if rtype == "rotamer-residue":
                C = _RotamerResidue
            elif rtype == "aa-residue":
                C = _Residue
            elif rtype == "residue":
                C = _Residue
            elif rtype == "ligand":
                C = _Ligand
            else:
                continue
            residue = C(
                self.data,
                selection=selection,
                parent=self,
                resi=resi,
                icode=icode,
                type=rtype,
            )
            self._residues.append(residue)

    def build_segments(self):
        if not self._residues:
            self.build_residues()

        segments = []
        segment = []
        for res in self._residues:
            if not segment:
                segment.append(res)
            else:
                prev = segment[-1]
                if prev.type == res.type:
                    bond_length = 10
                    if res.type in ("rotamer-residue", "aa-residue"):
                        # Check for nearness
                        sel = prev.select("name", "C")
                        C = prev._coor[sel]
                        sel = res.select("name", "N")
                        N = res._coor[sel]
                        bond_length = np.linalg.norm(N - C)
                    elif res.type == "residue":
                        # Check if RNA / DNA segment
                        O3 = prev.extract("name O3'")
                        P = res.extract("name P")
                        bond_length = np.linalg.norm(O3.coor[0] - P.coor[0])
                    if bond_length < 1.5:
                        segment.append(res)
                    else:
                        segments.append(segment)
                        segment = [res]
                else:
                    segments.append(segment)
                    segment = [res]

        segments.append(segment)

        for segment in segments:
            if len(segment) > 1:
                selections = [residue._selection for residue in segment]
                selection = np.concatenate(selections)
                segment = _Segment(
                    self.data, selection=selection, parent=self, residues=segment
                )
                self._segments.append(segment)


class _Segment(_BaseStructure):
    """Class that guarantees connected residues and allows
    backbone rotations."""

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.residues = kwargs["residues"]

    def __contains__(self, residue):
        for res in self.residues:
            if res.id == residue.id and res.altloc[0] == residue.altloc[0]:
                return True
        return False

    def __getitem__(self, arg):
        if isinstance(arg, int):
            return self.residues[arg]
        elif isinstance(arg, slice):
            residues = self.residues[arg]
            selections = []
            for residue in residues:
                selections.append(residue._selection)
            selection = np.concatenate(selections)
            return _Segment(
                self.data, selection=selection, parent=self.parent, residues=residues
            )
        else:
            raise TypeError

    def __len__(self):
        return len(self.residues)

    def __repr__(self):
        return "Segment: length {}".format(len(self.residues))

    def find(self, residue_id):
        if isinstance(residue_id, int):
            residue_id = (residue_id, "")
        for n, residue in enumerate(self.residues):
            if residue.id == residue_id:
                return n
        raise ValueError("Residue is not part of segment.")

    def rotate_psi(self, index, angle):
        """Rotate along psi dihedral (about the CA--C bond)."""
        selection = [residue._selection for residue in self.residues[index + 1 :]]
        residue = self.residues[index]
        selection.append(residue.select("name", ("O", "OXT")))
        selection = np.concatenate(selection)

        # Make an orthogonal axis system based on 3 atoms
        # TODO: Use .math.gram_schmidt_orthonormal_zx (or something similar)
        #       Note that here, the axes are 1→2, 2→1, 0=1×2.
        origin = system_coor[0].copy()
        CA = residue.extract("name", "CA").coor[0]
        C = residue.extract("name", "C").coor[0]
        O = residue.extract("name", "O").coor[0]
        system_coor = np.vstack((CA, C, O))
        system_coor -= origin
        zaxis = system_coor[1] / np.linalg.norm(system_coor[1])
        yaxis = system_coor[2] - np.inner(system_coor[2], zaxis) * zaxis
        yaxis /= np.linalg.norm(yaxis)
        xaxis = np.cross(yaxis, zaxis)

        # Create transformation matrix
        backward = np.vstack((xaxis, yaxis, zaxis))
        forward = backward.T
        angle = np.deg2rad(angle)
        rotation = Rz(angle)
        R = forward @ rotation @ backward

        # Apply transformation
        coor = self._coor[selection]
        coor -= origin
        coor = np.dot(coor, R.T)
        coor += origin
        self._coor[selection] = coor


def calc_rmsd(coor_a, coor_b):
    """Determine root-mean-square distance between two structures.

    Args:
        coor_a (np.ndarray[(n_atoms, 3), dtype=np.float]):
            Coordinates for structure a.
        coor_b (np.ndarray[(n_atoms, 3), dtype=np.float]):
            Coordinates for structure b.

    Returns:
        np.float:
            Distance between two structures.
    """
    return np.sqrt(np.mean((coor_a - coor_b) ** 2))


class Ensemble:
    """Represents a collection of Structure objects."""

    def __init__(self, structures: list[Structure]):
        """Initialize the Ensemble with a list of structures.

        Args:
            structures (list[Structure]): A list of Structure objects.
        """
        if not all(isinstance(s, Structure) for s in structures):
            raise TypeError("All items in the list must be Structure objects.")
        self._structures = list(structures)

    def __len__(self):
        """Return the number of structures in the ensemble."""
        return len(self._structures)

    def __getitem__(self, index):
        """Get a structure from the ensemble by index."""
        return self._structures[index]

    def append(self, structure: Structure):
        """Add a structure to the ensemble."""
        if not isinstance(structure, Structure):
            raise TypeError("Can only append Structure objects.")
        self._structures.append(structure)

    def tofile(self, fname: str, use_auth: bool = False):
        """Write the ensemble to an mmCIF file.

        Parameters
        ----------
        fname : str
            Filename to write to.
        use_auth : bool, optional
            Use auth_ identifiers instead of label_ identifiers. Default False.
        """
        if not self._structures:
            raise ValueError("Cannot write an empty ensemble.")

        # mmCIFFile.write handles writing multiple models
        mmCIFFile.write(fname, self._structures, use_auth=use_auth)

    def __repr__(self):
        return f"Ensemble: {len(self._structures)} structures."

    def calc_pairwise_rmsd(self) -> np.ndarray:
        """Calculate RMSD between all pairs of structures in the ensemble.

        Returns
        -------
        np.ndarray
            Matrix of pairwise RMSDs, shape (n_structures, n_structures)
        """
        n_structures = len(self._structures)
        rmsd_matrix = np.zeros((n_structures, n_structures))
        
        for i in range(n_structures):
            for j in range(i+1, n_structures):
                rmsd = calc_rmsd(self._structures[i].coor, self._structures[j].coor)
                rmsd_matrix[i,j] = rmsd
                rmsd_matrix[j,i] = rmsd
                
        return rmsd_matrix

    def align_to_reference(self, reference: Structure):
        """Align all structures in the ensemble to a reference structure in place.

        Uses the Kabsch algorithm.

        Parameters
        ----------
        reference : Structure
            Reference structure to align to. The coordinates of the structures
            in the ensemble will be modified.
        """
        if not self._structures:
            return # Nothing to align

        ref_coords = reference.coor
        if ref_coords is None or ref_coords.size == 0:
            raise ValueError("Reference structure has no coordinates.")

        # Calculate centroid of the reference structure
        ref_centroid = np.mean(ref_coords, axis=0)
        ref_centered = ref_coords - ref_centroid

        for structure in self._structures:
            mobile_coords = structure.coor
            if mobile_coords is None or mobile_coords.size == 0:
                # Or raise an error, or skip? Skipping for now.
                print(f"Warning: Skipping structure {structure} due to missing coordinates.")
                continue

            if mobile_coords.shape != ref_coords.shape:
                raise ValueError(
                    f"Structure {structure} has a different shape "
                    f"({mobile_coords.shape}) than the reference "
                    f"({ref_coords.shape}). Cannot align."
                )

            # Calculate centroid of the mobile structure
            mobile_centroid = np.mean(mobile_coords, axis=0)
            mobile_centered = mobile_coords - mobile_centroid

            # Compute covariance matrix
            # H = P^T * Q where P = mobile_centered, Q = ref_centered
            H = mobile_centered.T @ ref_centered

            # Perform SVD
            try:
                U, S, Vt = np.linalg.svd(H)
            except np.linalg.LinAlgError:
                 print(f"Warning: SVD computation failed for structure {structure}. Skipping alignment.")
                 continue


            # Calculate rotation matrix
            R = Vt.T @ U.T

            # Check for reflection and correct if necessary
            if np.linalg.det(R) < 0:
                # Multiply the column of Vt corresponding to the smallest singular value by -1
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Apply rotation and translation
            # Rotated coords = R * mobile_centered^T -> transpose back -> add ref_centroid
            # Or: mobile_centered @ R.T + ref_centroid
            aligned_coords = mobile_centered @ R.T + ref_centroid

            # Update structure coordinates in place
            structure.coor = aligned_coords

    def filter_by_rmsd(self, threshold: float, reference: Structure) -> 'Ensemble':
        """Filter structures based on RMSD to reference structure.

        Parameters
        ----------
        threshold : float
            RMSD threshold in Angstroms
        reference : Structure, optional
            Reference structure to compare against. If None, uses average structure.

        Returns
        -------
        Ensemble
            New ensemble containing only structures within RMSD threshold
        """ 
        filtered_structures = []
        for structure in self._structures:
            rmsd = calc_rmsd(structure.coor, reference.coor)
            if rmsd <= threshold:
                filtered_structures.append(structure)
                
        return Ensemble(filtered_structures)

    def combine(self, other: 'Ensemble') -> 'Ensemble':
        """Combine this ensemble with another ensemble.

        Parameters
        ----------
        other : Ensemble
            Another ensemble to combine with

        Returns
        -------
        Ensemble
            New ensemble containing all structures from both ensembles
        """
        combined_structures = self._structures + other._structures
        return Ensemble(combined_structures)

    def get_unique_chains(self) -> set:
        """Get unique chain IDs across the ensemble.

        Returns
        -------
        set
            Set of unique chain IDs
        """
        chain_ids = set()
        for structure in self._structures:
            chain_ids.update(chain.id for chain in structure.chains)
        return chain_ids

    def normalize_occupancies(self):
        """Normalize occupancies across the ensemble.
        
        Sets the occupancy of each structure to 1/N where N is the number of structures
        in the ensemble. This ensures that the sum of occupancies across all ensemble
        members equals 1.0 for each residue.
        """
        if not self._structures:
            raise ValueError("Cannot normalize occupancies of empty ensemble")
            
        occupancy = 1.0 / len(self._structures)
        
        for structure in self._structures:
            structure.q[:] = occupancy

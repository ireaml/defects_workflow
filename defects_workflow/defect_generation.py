"""Functions to generate defects from a given structure."""

from __future__ import annotations
import numpy as np
import warnings
from copy import deepcopy

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import BadIncarWarning, incar_params
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.defects.generators import (
    VacancyGenerator,
    AntiSiteGenerator,
    VoronoiInterstitialGenerator,
    _remove_oxidation_states,
)
from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.thermo import DefectEntry

from aiida.orm import Dict, Float, Int, Bool, Str, List, StructureData, ArrayData
from aiida.engine import calcfunction, workfunction

from shakenbreak.input import _get_defect_name_from_obj

from defects_workflow.charge_tools import get_charges, group_ions, extend_list_to_zero


def _generate_defects(
    bulk: Structure | StructureData,
    symprec: float=0.01,
    angle_tolerance: float=5,
    interstitial_min_dist: float=0.9,
    defect_types: list[str]=["vacancies", "antisites", "interstitials"],
):
    """
    Generate all intrinsic defects for a given conventional or primitive
    structure.

    Args:
    bulk (Structure | StructureData):
        Structure to generate defects for.
    symprec (float, optional):
        Symmetry precision. Defaults to 0.01.
    angle_tolerance (float, optional):
        Angle tolerance. Defaults to 5.
    interstitial_min_dist (float, optional):
        Minimum distance between interstitial atom and other atoms.
    defect_type (list[str], optional):
        List of defect types to generate. Defaults to ["vacancies", "antisites", "interstitials"].

    Returns:
        defect_dict (dict): Dictionary of Defects, with keys "vacancies", "antisites"
            and "interstitials". Each key contains a dictionary of Defect objects:
            {"vacancies": {defect_name: Defect, ...}, "antisites": {...}, "interstitials": {...},
    """
    # Check bulk primitive structure:
    if isinstance(bulk, StructureData):
        bulk = bulk.get_pymatgen_structure()
    prim = bulk.get_primitive_structure()
    prim_no_oxi = _remove_oxidation_states(prim)
    # Generate defects:
    vac_gen = VacancyGenerator(symprec=symprec, angle_tolerance=angle_tolerance,)
    ant_gen = AntiSiteGenerator(symprec=symprec, angle_tolerance=angle_tolerance,)
    int_gen = VoronoiInterstitialGenerator(min_dist=interstitial_min_dist, angle_tol=angle_tolerance,)
    # These are lists of Defect objects:
    defects_dict = {}
    if "vacancies" in defect_types:
        defects_dict["vacancies"] = vac_gen.get_defects(structure=prim_no_oxi.copy())
    if "antisites" in defect_types:
        defects_dict["antisites"] = ant_gen.get_defects(structure=prim_no_oxi.copy())
    if "interstitials" in defect_types:
        defects_dict["interstitials"] = int_gen.get_defects(
            structure=prim_no_oxi.copy(),
            insert_species=[*map(str, prim.composition.elements)]
        )
    for defect_type, defect_list in defects_dict.items():
        defects_dict[defect_type] = {
            _get_defect_name_from_obj(defect): defect for defect in defect_list
        }
    return defects_dict


def add_charge_states(
    defect_dict: dict,
    charge_tolerance: float=5,
):
    """Specify the charge states for each defect, storing them under the
    .user_charges attibute.
    These are determined by considering the abundance of each oxidation state for
    the defect element. The charge state are only considered if they represent greater
    than `charge_tolerance`% of all the oxidation states of the atoms in the ICSD
    ICSD data taken from: `doi.org/10.1021/acs.jpclett.0c02072`

    Args:
    defect_dict (dict):
        Dictionary of Defects, with format:
        {"vacancies": [Defect, ...], "interstitials": [...], "antisites": [],}
    charge_tolerance (float, optional):
        Tolerance for charge states. Defaults to 5(%).

    Returns:
    defect_dict (dict):
        Dictionary of defects with charge states specified in .user_charges attribute.
        The format of the dictionary is:
        {"vacancies": {defect_name: Defect, ...}, "antisites": {...}, "interstitials": {...},
    """
    def _get_antisite_charges(site, sub, structure) -> list[int]:
        elements = [site.species.elements[0].symbol for site in structure]
        cations = group_ions(elements, "cation", 5)
        anions = group_ions(elements, "anion", 5)
        site_charges = get_charges(site, charge_tolerance) * -1
        sub_charges = get_charges(sub, charge_tolerance)

        if sub in cations and site in cations:
            charges = list(
                np.arange(
                    max(sub_charges) + min(site_charges),
                    max(sub_charges) + 1,
                    1,
                    dtype=int,
                )
            )
            if 0 not in charges:
                charges = extend_list_to_zero(charges)
            return sorted(charges)
        elif sub in anions and site in anions:
            charges = np.arange(
                min(sub_charges) + min(site_charges), min(sub_charges) + 1, 1, dtype=int
            )
            if 0 not in charges:
                charges = extend_list_to_zero(charges)
            return sorted(charges)

    for defect_type, defect_subdict in defect_dict.items():
        if defect_type in ["vacancies", "interstitials"]:
            for defect in defect_subdict.values():
                element = defect.defect_site.species.elements[0].symbol
                selected_charges = get_charges(
                    atom=element,
                    charge_tol=charge_tolerance
                )
                defect.user_charges = selected_charges
        elif defect_type  == "antisites":
            # For antisites, charges depend on the original/bulk and
            # substitution site
            for defect in defect_subdict.values():
                original_symbol = defect.defect_site.species.elements[0].symbol
                sub_symbol = defect.site.species.elements[0].symbol
                defect.user_charges = _get_antisite_charges(
                    site=original_symbol, sub=sub_symbol, structure=defect.structure
                )
        else:
            raise ValueError(f"Defect type {defect_type} not recognised.")
    return defect_dict


def generate_defect_entries(
    defects_dict: dict,
    sc_mat: np.ndarray | None = None,
    dummy_species: str | None = None,
    min_atoms: int = 80,
    max_atoms: int = 140,
    min_length: float = 10,
    force_diagonal: bool = False,
):
    """
    Generate a DefectEntry for each charge state of the Defect.
    Set up supercell for each DefectEntry (accessible as DefectEntry.sc_entry.structure)
    and charge state (accessible as DefectEntry.charge_state).

    Args:
    defects_dict (dict):
        Dictionary of Defects, with keys "vacancies", "antisites" and "interstitials".
    sc_mat (np.ndarray, optional):
    dummy_species (str, optional):
    min_atoms (int, optional):
        Minimum number of atoms in supercell. Defaults to 80.
    max_atoms (int, optional):
        Maximum number of atoms in supercell. Defaults to 140.
    min_length (float, optional):
        Minimum length of supercell. Defaults to 10.
    force_diagonal (bool, optional):

    Returns:
    defects_dict (dict):
        Dictionary of DefectEntries, with keys "vacancies", "antisites"
        and "interstitials".
    """
    def get_defect_entry_from_defect(
        defect: Defect,
        charge_state: int,
        supercell_struct: Structure,
    ):
        """Assuming defect supercell has been generated with dummy atom"""
        # Get defect frac coords in supercell
        sc_defect_frac_coords = supercell_struct.pop(-1).frac_coords  # Added at the end
        sc_entry = ComputedStructureEntry(
            structure=supercell_struct,
            energy=0.0,  # Needs to be set, so just set to 0.0
        )
        return DefectEntry(
            defect=defect,
            charge_state=charge_state,
            sc_entry=sc_entry,
            sc_defect_frac_coords=sc_defect_frac_coords,
        )

    if not dummy_species:
        raise ValueError(
            "dummy_species must be specified! This is used to keep track"
            " of the defect coordinates in the supercell."
        )
    for key, value in defects_dict.items():  # for defect type (e.g. vacancies)
        defect_entries = {}
        for defect_name, defect in value.items():
            supercell_struct = defect.get_supercell_structure(
                sc_mat=sc_mat,
                dummy_species=dummy_species,  # for vacancies, to get defect frac coords in supercell
                max_atoms=max_atoms,
                min_atoms=min_atoms,
                min_length=min_length,
                force_diagonal=force_diagonal,
            )
            defect_entries[defect_name] = []
            for charge_state in defect.user_charges:
                defect_entries[defect_name].append(
                    get_defect_entry_from_defect(
                        defect=defect,
                        charge_state=charge_state,
                        supercell_struct=supercell_struct,
                    )
                )
        defects_dict[key] = deepcopy(defect_entries)  # for each defect type, dict of DefectEntries
    return defects_dict


@calcfunction
def setup_defects(
    bulk: StructureData,
    defect_types: List[str]=List(['vacancies', 'interstitials', 'antisites']),
    symprec: Float=Float(0.01),
    angle_tolerance: Int=Int(5),
    interstitial_min_dist: Float=Float(1.0),
) -> Dict:
    """Generate defects, add defect charge states and setup supercell.

    Args:
        bulk (StructureData):
            Bulk structure.
        defect_types (List[str], optional):
            List of defect types to generate.
            Defaults to List(['vacancies', 'interstitials', 'antisites']).
        symprec (Float, optional):
            Symmetry precision. Defaults to Float(0.01).
        angle_tolerance (Int, optional):
            Angle tolerance. Defaults to Int(5).
        interstitial_min_dist (Float, optional):
            Minimum distance for interstitials. Defaults to Float(0.9).

    Returns:
        Dict: Dictionary of Defects, with keys "vacancies", "antisites" and
            "interstitials", e.g.:
            {"vacancies": [defect_name: [Defect, Defect,] ...],
            "antitists": [defect_name: [Defect, Defect,], ...], ...}
    """
    defects_dict = _generate_defects(
        bulk.get_pymatgen_structure(),
        symprec.value,
        angle_tolerance.value,
        interstitial_min_dist.value,
        defect_types.get_list(),
    )
    return Dict(dict=defects_dict)


@calcfunction
def setup_defect_charge_states(
    defects_dict: Dict,
    charge_tolerance: Int=Int(5),
) -> Dict:
    """Add charge states, based on common element oxi states"""
    defects_dict = add_charge_states(
        defect_dict=defects_dict.get_dict(),
        charge_tolerance=charge_tolerance.value,
    )
    return Dict(dict=defects_dict)


@calcfunction
def setup_defect_supercells(
    defects_dict: Dict,
    sc_mat: ArrayData | None = None,
    dummy_species: Str | None = Str("X"),
    min_atoms: Int = Int(30),
    max_atoms: Int = Int(150),
    min_length: Float = Float(10),
    force_diagonal: Bool = Bool(False),
) -> Dict:
    """Generate supercell defect structure, and refactor Defect to DefectEntry objects

    Args:
        sc_mat (ArrayData, optional):
            Supercell matrix. Defaults to None.
        dummy_species (Str, optional):
            Dummy species. Defaults to None.
        min_atoms (Int, optional):
            Minimum number of atoms in supercell. Defaults to Int(80).
        max_atoms (Int, optional):
            Maximum number of atoms in supercell. Defaults to Int(140).
        min_length (Float, optional):
            Minimum length of supercell. Defaults to Float(10).
        force_diagonal (Bool, optional):
            Force diagonal supercell. Defaults to Bool(False

    Returns:
        Dict: Dictionary of DefectEntry's, with keys "vacancies", "antisites" and
            "interstitials", e.g.:
            {"vacancies": [defect_name: [DefectEntry, DefectEntry,] ...],
            "antitists": [defect_name: [DefectEntry, DefectEntry,], ...], ...}
    """
    defect_entries_dict = generate_defect_entries(
        defects_dict=defects_dict.get_dict(),
        sc_mat=sc_mat.get_array(),
        dummy_species=dummy_species.value,
        min_atoms=min_atoms.value,
        max_atoms=max_atoms.value,
        min_length=min_length.value,
        force_diagonal=force_diagonal.value,
    )
    return Dict(dict=defect_entries_dict)


@calcfunction
def sort_interstitials_for_screening(defects_dict: Dict):
    """
    Loop over defects_dict to identify interstitials of the
    same element, and select them for screening.

    (These will be relaxed to find the most stable interstitial/
    avoid cases where different initial structures lead to the
    same final structure.)
    """
    # Group interstitials by element:
    dict_interstitials = {}  # {Te_i: {Te_i_s32: DefectEntry, ...}, Cd_i: {...}}}
    for defect_name, entry in defects_dict.value["interstitials"].items():
        if entry.charge_state == 0: # only neutral for screening
            if entry.defect.name in dict_interstitials:
                dict_interstitials[entry.defect.name][defect_name] = entry
            else:
                dict_interstitials[entry.defect.name] = {defect_name: entry}
    # Select cases with more than one interstitial:
    dict_interstitials = {
        k: v for k, v in dict_interstitials.items() if len(v) > 1
    }
    return Dict(dict=dict_interstitials)
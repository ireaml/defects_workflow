"""Functions to generate defects from a given structure."""

from __future__ import annotations
import numpy as np
import warnings
from copy import deepcopy

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import BadIncarWarning, incar_params
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.defects.generators import (
    VacancyGenerator, AntiSiteGenerator, VoronoiInterstitialGenerator,
    _remove_oxidation_states,
)
from pymatgen.analysis.defects.thermo import DefectEntry

from aiida.orm import Dict, Float, Int, Bool, Str, StructureData, ArrayData
from aiida.engine import calcfunction, workfunction

from charge_tools import get_charges, group_ions, extend_list_to_zero


def generate_defects(
    bulk: Structure | StructureData,
    symprec: float=0.01,
    angle_tolerance: float=5,
    interstitial_min_dist: float=0.9,
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

    Returns:
        defect_dict (dict): Dictionary of defects, with keys "vacancies", "antisites"
            and "interstitials".
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
    vacancies = vac_gen.get_defects(structure=prim_no_oxi.copy())
    antisites = ant_gen.get_defects(structure=prim_no_oxi.copy())
    interstitials = int_gen.get_defects(
        structure=prim_no_oxi.copy(),
        insert_species=[*map(str, prim.composition.elements)]
    )
    return {
        "vacancies": vacancies,
        "antisites": antisites,
        "interstitials": interstitials
    }


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
        Dictionary of defects.
    charge_tolerance (float, optional):
        Tolerance for charge states. Defaults to 5(%).

    Returns:
    defect_dict (dict):
        Dictionary of defects with charge states specified in .user_charges attribute.
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

    for defect_type, defect_list in defect_dict.items():
        if defect_type in ["Vacancy", "Interstitial"]:
            for defect in defect_list:
                element = defect.defect_site.species.elements[0].symbol
                selected_charges = get_charges(
                    atom=element,
                    charge_tol=charge_tolerance
                )
                for defect in defect_dict[defect_type]:
                    defect.user_charges = selected_charges
        elif defect_type  == "antisites":
            # For antisites, charges depend on the original/bulk and
            # substitution site
            original_symbol = defect.defect_site.species.elements[0].symbol
            sub_symbol = defect.site.species.elements[0].symbol
            defect.user_charges = _get_antisite_charges(
                site=original_symbol, sub=sub_symbol, structure=defect.structure
            )
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
    for key, value in defects_dict.items():
        defect_entries = []
        for defect in value:
            supercell_struct = defect.get_supercell_structure(
                sc_mat=sc_mat,
                dummy_species="X",  # for vacancies, to get defect frac coords in supercell
                max_atoms=max_atoms,
                min_atoms=min_atoms,
                min_length=min_length,
                force_diagonal=force_diagonal,
            )
            # Get defect frac coords in supercell
            sc_defect_frac_coords = supercell_struct.pop(-1).frac_coords  # Added at the end

            sc_entry = ComputedStructureEntry(
                structure=supercell_struct,
                energy=0.0,  # Needs to be set, so just set to 0.0
            )
            for charge_state in defect.user_charges:
                defect_entries.append(
                    DefectEntry(
                        defect=defect,
                        charge_state=charge_state,
                        sc_entry=sc_entry,
                        sc_defect_frac_coords=sc_defect_frac_coords,
                    )
                )
        defects_dict[key] = defect_entries  # for each defect, list of DefectEntries
    return defects_dict


@calcfunction
def generate_supercell_n_defects(
    bulk: StructureData,
    symprec: Float=Float(0.01),
    angle_tolerance: Int=Int(5),
    interstitial_min_dist: Float=Float(0.9),
    sc_mat: ArrayData | None = None,
    dummy_species: Str | None = None,
    min_atoms: Int = Int(80),
    max_atoms: Int = Int(140),
    min_length: Float = Float(10),
    force_diagonal: Bool = Bool(False),
):
    """Generate defects and setup supercell.

    Args:
        bulk (StructureData):
            Bulk structure.
        symprec (Float, optional):
            Symmetry precision. Defaults to Float(0.01).
        angle_tolerance (Int, optional):
            Angle tolerance. Defaults to Int(5).
        interstitial_min_dist (Float, optional):
            Minimum distance for interstitials. Defaults to Float(0.9).
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
    """
    defects_dict = generate_defects(
        bulk.get_pymatgen_structure(),
        symprec,
        angle_tolerance,
        interstitial_min_dist
    )
    defect_entries_dict = generate_defect_entries(
        defects_dict=defects_dict,
        sc_mat=sc_mat,
        dummy_species=dummy_species,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        min_length=min_length,
        force_diagonal=force_diagonal,
    )
    return Dict(dict=defect_entries_dict)


"""Functions to generate defects from a given structure."""

from __future__ import annotations
import numpy as np
import warnings

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import BadIncarWarning, incar_params
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.defects.generators import (
    VacancyGenerator, AntiSiteGenerator, VoronoiInterstitialGenerator, _remove_oxidation_states,
)
from pymatgen.analysis.defects.thermo import DefectEntry

from aiida.orm import Dict, Float, Int, Bool, StructureData, ArrayData
from aiida.engine import calcfunction, workfunction

from shakenbreak.input import Distortions
from shakenbreak.vasp import default_incar_settings, _scaled_ediff

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
                dummy_species=dummy_species,
                max_atoms=max_atoms,
                min_atoms=min_atoms,
                min_length=min_length,
                force_diagonal=force_diagonal,
            )
            sc_entry = ComputedStructureEntry(
                structure=supercell_struct,
                energy=None,  # Needs to be set, so just set to None
            )
            for charge_state in defect.user_charges:
                defect_entry = DefectEntry(
                    defect=defect,
                    charge_state=charge_state,
                    sc_entry=sc_entry,
                )
                defect_entries.append(defect_entry)
        defects_dict[key] = defect_entries
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


def setup_incar_snb(
    single_defect_dict: dict,
    input_dir: str = None,
    incar_settings: dict = None,
    potcar_settings: dict = None,
) -> None:
    """
    Generates input files for vasp Gamma-point-only relaxation.
    Args:
        single_defect_dict (:obj:`dict`):
            Single defect-dictionary from prepare_vasp_defect_inputs()
            output dictionary of defect calculations (see example notebook)
        input_dir (:obj:`str`):
            Folder in which to create vasp_gam calculation inputs folder
            (Recommended to set as the key of the prepare_vasp_defect_inputs()
            output directory)
            (default: None)
        incar_settings (:obj:`dict`):
            Dictionary of user INCAR settings (AEXX, NCORE etc.) to override
            default settings. Highly recommended to look at
            `/SnB_input_files/incar.yaml`, or output INCARs or doped.vasp_input
            source code, to see what the default INCAR settings are.
            (default: None)
        potcar_settings (:obj:`dict`):
            Dictionary of user POTCAR settings to override default settings.
            Highly recommended to look at `default_potcar_dict` from
            doped.vasp_input to see what the (Pymatgen) syntax and doped
            default settings are.
            (default: None)
    Returns:
        None
    """
    supercell = single_defect_dict["Defect Structure"]
    num_elements = len(supercell.composition.elements)  # for ROPT setting in INCAR

    warnings.filterwarnings(
        "ignore", category=BadInputSetWarning
    )  # Ignore POTCAR warnings because Pymatgen incorrectly detecting POTCAR types
    potcar_dict = deepcopy(default_potcar_dict)
    if potcar_settings:
        if "POTCAR_FUNCTIONAL" in potcar_settings.keys():
            potcar_dict["POTCAR_FUNCTIONAL"] = potcar_settings["POTCAR_FUNCTIONAL"]
        if "POTCAR" in potcar_settings.keys():
            potcar_dict["POTCAR"].update(potcar_settings.pop("POTCAR"))

    defect_relax_set = DefectRelaxSet(
        supercell,
        charge=single_defect_dict["Transformation Dict"]["charge"],
        user_potcar_settings=potcar_dict["POTCAR"],
        user_potcar_functional=potcar_dict["POTCAR_FUNCTIONAL"],
    )
    potcars = _check_psp_dir()

    relax_set_incar = defect_relax_set.incar
    try:
        # Only set if change in NELECT
        nelect = relax_set_incar.as_dict()["NELECT"]
    except KeyError:
        # Get NELECT if no change (-dNELECT = 0)
        nelect = defect_relax_set.nelect

    # Update system dependent parameters
    default_incar_settings_copy = default_incar_settings.copy()
    default_incar_settings_copy.update(
        {
            "NELECT": nelect,
            "NUPDOWN": f"{nelect % 2:.0f} # But could be {nelect % 2 + 2:.0f} "
            + "if strong spin polarisation or magnetic behaviour present",
            "EDIFF": f"{_scaled_ediff(supercell.num_sites)} # May need to reduce for tricky relaxations",
            "ROPT": ("1e-3 " * num_elements).rstrip(),
        }
    )
    if incar_settings:
        for (
            k
        ) in (
            incar_settings.keys()
        ):  # check INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    f"Cannot find {k} from your incar_settings in the list of "
                    "INCAR flags",
                    BadIncarWarning,
                )
        default_incar_settings_copy.update(incar_settings)

    return default_incar_settings_copy


def apply_shakenbreak(
    defects_dict: dict,
    distortion_increment: Float=Float(0.1),
    stdev: Float=Float(0.25),
):
    # Refactor defects_dict to be a list of defects
    defects = sum(defects_dict.values(), [])
    dist = Distortions(
        defects=defects,
        distortion_increment=distortion_increment,
        stdev=stdev,
    )
    # Now we need to get INCAR dict for each defect
    # Need to homogenize potcar keywords of setup_incar_snb
    # and the potcar keywords of the workchain
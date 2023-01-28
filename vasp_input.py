"""Functions to setup vasp inputs"""
import os
import warnings
from copy import deepcopy

from monty.serialization import loadfn
from pymatgen.io.vasp.sets import BadInputSetWarning
from pymatgen.io.vasp.inputs import BadIncarWarning, incar_params
from pymatgen.analysis.defects.thermo import DefectEntry

from shakenbreak.vasp import _scaled_ediff, _check_psp_dir, DefectRelaxSet

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(f"{MODULE_DIR}/yaml_files/vasp/default_POTCARs.yaml")
# Load default INCAR settings for the ShakenBreak geometry relaxations
default_snb_incar_settings = loadfn(
    os.path.join(MODULE_DIR, "yaml_files/vasp/incar/relax_SnB.yaml")
)

# TODO: refactor for pymatgen.analysis.defects
def setup_incar_snb(
    defect_entry: DefectEntry,
    input_dir: str = None,
    incar_settings: dict = None,
    potcar_settings: dict = None,
) -> None:
    """
    Generates input files for vasp Gamma-point-only relaxation.

    Args:
        single_defect_dict (:obj:`DefectEntry`):
            pymatgen.analysis.defects.thermo.DefectEntry object
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
        DefectRelaxSet object
    """
    supercell = defect_entry.sc_entry.structure
    num_elements = len(supercell.composition.elements)  # for ROPT setting in INCAR
    charge = defect_entry.charge_state

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
        charge=charge,
        user_potcar_settings=potcar_dict["POTCAR"],
        user_potcar_functional=potcar_dict["POTCAR_FUNCTIONAL"],
    )
    # Check POTCARs are present
    potcars = _check_psp_dir()
    if not potcars:
        raise ValueError(
            "POTCAR directory not set up with pymatgen. "
            "POTCARs are needed to determine appropriate "
            "NELECT setting in INCAR files. Exiting!"
        )

    relax_set_incar = defect_relax_set.incar
    try:
        # Only set if change in NELECT
        nelect = relax_set_incar.as_dict()["NELECT"]
    except KeyError:
        # Get NELECT if no change (-dNELECT = 0)
        nelect = defect_relax_set.nelect

    # Update system dependent parameters
    default_snb_incar_settings_copy = default_snb_incar_settings.copy()
    default_snb_incar_settings_copy.update(
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
        ):  # check user INCAR flags and warn if they don't exist (typos)
            if (
                k not in incar_params.keys()
            ):  # this code is taken from pymatgen.io.vasp.inputs
                warnings.warn(  # but only checking keys, not values so we can add comments etc
                    f"Cannot find {k} from your incar_settings in the list of "
                    "INCAR flags",
                    BadIncarWarning,
                )
        default_snb_incar_settings_copy.update(incar_settings)

    defect_relax_set.incar = default_snb_incar_settings_copy
    return defect_relax_set
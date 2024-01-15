import os
from copy import deepcopy

from typing import Optional
from typing import Union

from monty.serialization import dumpfn, loadfn
import warnings

# pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar

# aiida
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import WorkChain, calcfunction, submit
from aiida.orm import Dict, Str, Bool, Int, StructureData, KpointsData, Code
from aiida.orm.nodes.data.remote.base import RemoteData
from aiida.common.extendeddicts import AttributeDict
from aiida.tools.groups import GroupPath

# aiida-vasp
from aiida_vasp.data.chargedensity import ChargedensityData
from aiida_vasp.data.wavefun import WavefunData

from defects_workflow.vasp_toolkit.input import get_potcar_mapping

path = GroupPath()

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_incar_settings = loadfn(
    os.path.join(MODULE_DIR, "yaml_files/vasp/incar/relax_host.yaml")
)
default_potcar_dict = loadfn(
    os.path.join(MODULE_DIR, "yaml_files/vasp/default_POTCARs.yaml")
)


def setup_relax_inputs(
    code_string: str,
    # VASP Input:
    structure_data: Union[StructureData, Structure],
    kpoints_data: KpointsData,
    # Aiida parameters:
    options: dict,
    settings: Optional[dict] = None,
    metadata: Optional[dict] = None,
    clean_workdir: Optional[bool] = False,
    # Labels:
    workchain_label: Optional[str]=None,
    # INCAR related parameters:
    incar_dict: Optional[dict] = None,
    use_default_incar_settings: Optional[bool] = False,
    positions: bool = True,
    shape: bool = False,
    volume: bool = False,
    ionic_steps: int = 300,
    algo: Optional[str] = "cg",  # Optimization algorithm, default to conjugate gradient
    dynamics: Optional[dict] = None,
    # POTCAR parameters:
    potcar_family: Optional[str] = "GGA_PAW_PBE", #"PAW_PBE_54",  # In mac name is GGA_PAW_PBE
    potential_mapping: Optional[dict] = None,
    # Restart:
    chgcar: Optional[ChargedensityData] = None,
    wavecar: Optional[WavefunData] = None,
    remote_folder: Optional[RemoteData] = None,
):
    """Setup and submit relaxation workchain"""
    def setup_settings(settings: dict=None):
        default_settings = {
            "parser_settings": {
                "misc": [
                    "total_energies",
                    "maximum_force",
                    "maximum_stress",
                    "run_status",
                    "run_stats",
                    "notifications",
                ],
                "add_structure": True,  # retrieve structure and kpoints
                "add_energies": True,
                # Dont parse:
                'add_kpoints': False,
                'add_forces' : False,
                'add_bands': False,
                'add_trajectory': False,
            },
        }
        if settings:
            default_settings.update(settings)
        return Dict(dict=default_settings)

    def setup_structure(pmg_structure):
        sorted_structure = pmg_structure.get_sorted_structure()
        if sorted_structure != pmg_structure:
            print(
                "Structure is not sorted, sorting now. "
                +"\n Quit if you had set MAGMOM's or selective dynamics for the unsorted structure!"
            )
        return StructureData(pymatgen=sorted_structure)

    def setup_potential(inputs, pmg_structure, potcar_family, potential_mapping):
        inputs.potential_family = Str(potcar_family)
        if potential_mapping:
            inputs.potential_mapping = Dict(dict=potential_mapping)
        else:
            inputs.potential_mapping = Dict(
                dict=get_potcar_mapping(structure=pmg_structure)
        )
        return inputs

    def setup_incar(
        inputs,
        default_incar_settings,
        incar_dict,
        use_default_incar_settings,
        positions,
        shape,
        volume,
        algo,
        ionic_steps
    ):
        if not incar_dict:
            warnings.warn(
                "User didn't provide `incar_dict`, so the default INCAR settings are used!"
            )
            default_incar_settings_copy = deepcopy(default_incar_settings)
        if use_default_incar_settings and incar_dict:
            default_incar_settings_copy = deepcopy(default_incar_settings)
            default_incar_settings_copy.update(incar_dict)
        elif not use_default_incar_settings and incar_dict:
            default_incar_settings_copy = deepcopy(incar_dict)

        # Check no typos in keys
        incar = Incar(default_incar_settings_copy)
        incar.check_params()  # check keys
        incar_dict = incar.as_dict()
        del incar_dict["@class"]
        del incar_dict["@module"]

        # Check ICHARG tag
        if incar_dict.get("ICHARG") != 2:
            if (incar_dict.get("ICHARG") in [1, 11]) and (
                not remote_folder
            ):  # needs CHGCAR as input, so make sure we're giving it
                assert chgcar is not None, "CHGCAR is required for ICHARG = 1 or 11"
            if (incar_dict.get("ICHARG") == 0) and (
                not remote_folder
            ):  # needs WAVECAR as input, so make sure we're giving it
                assert wavecar is not None, "WAVECAR is required for ICHARG = 0"
        inputs.parameters = Dict(dict={"incar": incar_dict})

        # Relaxation related parameters that is passed to the relax workchain
        relax = AttributeDict()
        # Turn on relaxation
        relax.perform = Bool(True)
        # Select relaxation algorithm
        relax.algo = Str(algo)
        # Set force cutoff limit (EDIFFG, but no sign needed)
        force_cutoff = abs(default_incar_settings_copy.get("EDIFFG", 0.01))
        relax.force_cutoff = DataFactory("float")(force_cutoff)


        # Turn on relaxation of positions (strictly not needed as the default is on)
        # The three next parameters correspond to the well known ISIF=3 setting
        relax.positions = Bool(positions)
        # Relaxation of the cell shape (defaults to False)
        relax.shape = Bool(shape)
        # Relaxation of the volume (defaults to False)
        relax.volume = Bool(volume)
        # Set maximum number of ionic steps
        relax.steps = Int(ionic_steps)
        # Set the relaxation parameters on the inputs
        inputs.relax = relax
        return inputs

    def setup_metadata(metadata=None, workchain_label=None, pmg_structure=None):
        if not metadata:
            if not workchain_label:
                formula = pmg_structure.composition.to_pretty_string()
                workchain_label = f"Relax_{formula}"
            metadata = {"label": workchain_label}
        return metadata

    def setup_restart_files(inputs, wavecar, chgcar, remote_folder):
        if chgcar:
            inputs.chgcar = chgcar
        if wavecar:
            inputs.wavecar = wavecar
        if remote_folder:
            inputs.restart_folder = remote_folder
        return inputs

    # We set the workchain you would like to call
    workchain = WorkflowFactory("vasp.relax")

    # We declare the input containers
    inputs = AttributeDict()

    # Set inputs for the following WorkChain execution

    # Set code
    inputs.code = Code.get_from_string(code_string)

    # Set structure
    if isinstance(structure_data, StructureData):
        pmg_structure = structure_data.get_pymatgen_structure()
    elif isinstance(structure_data, Structure):
        pmg_structure = structure_data
    inputs.structure = setup_structure(pmg_structure)

    # Set k-points grid density
    inputs.kpoints = kpoints_data

    # Set INCAR parameters
    inputs = setup_incar(
        inputs,
        default_incar_settings,
        incar_dict,
        use_default_incar_settings,
        positions,
        shape,
        volume,
        algo,
        ionic_steps
    )
    # Set potentials and their mapping
    inputs = setup_potential(
        inputs, pmg_structure, potcar_family, potential_mapping
    )

    # Set dynamics
    if dynamics:
        inputs.dynamics = dynamics

    # Set options
    inputs.options = Dict(dict=options)

    # Set settings
    inputs.settings = setup_settings(settings=settings)

    # Metadata
    inputs.metadata = setup_metadata(metadata, workchain_label, pmg_structure)

    # Set workchain related inputs, in this case, give more explicit output to report
    inputs.verbose = Bool(True)

    # Chgcar and wavecar
    inputs = setup_restart_files(inputs, wavecar, chgcar, remote_folder)

    # Clean Workdir
    # If True, clean the work dir upon the completion of a successfull calculation.
    inputs.clean_workdir = Bool(clean_workdir)

    return workchain, inputs


# @calcfunction
# def wrapper_submit_relaxation(
#     code_string: Str,
#     structure_data: StructureData,
#     kpoints_data: KpointsData,
#     options: Dict,
#     # Labels:
#     workchain_label: Optional[Str] = None,
#     group_label: Optional[Str] = None,
#     # INCAR related parameters:
#     incar_dict: Optional[Dict] = None,
#     use_default_incar_settings: Optional[Bool] = Bool(False),
#     positions: Bool = Bool(True),
#     shape: Bool = Bool(False),
#     volume: Bool = Bool(False),
#     ionic_steps: Int = Int(300),
#     algo: Optional[Str] = Str("cg"),  # Optimization algorithm, default to conjugate gradient
#     dynamics: Optional[Dict] = None,
#     # POTCAR parameters:
#     potcar_family: Optional[Str] = Str("PAW_PBE_54"),
#     potential_mapping: Optional[Dict] = None,
#     # Other parameters:
#     settings: Optional[Dict] = None,
#     metadata: Optional[Dict] = None,
#     # Optional restart files:
#     chgcar: Optional[ChargedensityData] = None,
#     wavecar: Optional[WavefunData] = None,
#     remote_folder: Optional[RemoteData] = None,
#     clean_workdir: Optional[Bool] = Bool(False),
# ):
#     """Aiida wrapper for submit_relxation (where input args are aiida.orm data types)
#     """
#     workchain = submit_relaxation(
#         code_string=code_string.value,
#         structure_data=structure_data,
#         kpoints_data=kpoints_data,
#         # Labels:
#         workchain_label=workchain_label.value if workchain_label else None,
#         group_label=group_label.value if group_label else None,
#         # Relaxation inputs:
#         incar_dict=incar_dict.value if incar_dict else None,
#         use_default_incar_settings=use_default_incar_settings.value,
#         positions=positions.value,
#         shape=shape.value,
#         volume=volume.value,
#         ionic_steps=ionic_steps.value,
#         algo=algo.value if algo else None,
#         dynamics=dynamics.value if dynamics else None,
#         potcar_family=potcar_family.value if potcar_family else None,
#         potential_mapping=potential_mapping.value if potential_mapping else None,
#         # Aiida related tags:
#         options=dict(options),
#         settings=dict(settings) if settings else None,
#         metadata=dict(metadata) if metadata else None,
#         # Restart files:
#         chgcar=chgcar,
#         wavecar=wavecar,
#         remote_folder=remote_folder,
#         clean_workdir=clean_workdir.value,
#     )
#     return workchain
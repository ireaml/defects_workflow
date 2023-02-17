"""Miscellaneous functions & calcfunctions used within the workflow."""

from copy import deepcopy
import os
from monty.serialization import loadfn

# pymatgen
#from pymatgen.ext.matproj import MPRester
from mp_api.client import MPRester
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# aiida
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Dict, Float, Str, StructureData, KpointsData, Code
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import submit, calcfunction

# Parameters:
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_gamma_kpoints():
    """Setup Gamma point kpoints"""
    gam_kpts = KpointsData()
    gam_kpts.set_kpoints_mesh([1, 1, 1])
    return gam_kpts


@calcfunction
def query_materials_project(
    api_key: Str,
    material_id: Str,
) -> StructureData:
    """Query material from materials project database based on mp-id"""
    mpr = MPRester(api_key=api_key.value)
    data = mpr.summary.search(
        material_ids=[material_id.value,],
        fields=["structure"]
    )
    # Get primitive cell
    prim = data[0].structure.get_primitive_structure()
    return StructureData(pymatgen_structure=prim)


@calcfunction
def get_kpoints_from_density(
    structure: StructureData,
    k_density: Float=Float(900),
) -> KpointsData:
    """Calculate the kpoints required based on kpoint density"""
    pmg_structure = structure.get_pymatgen_structure()
    kpoints = Kpoints.automatic_density(
        structure=pmg_structure,
        kppa=k_density.value
    )
    kpoints_data = KpointsData()
    kpoints_data.set_kpoints_mesh(kpoints.kpts[0])
    return kpoints_data


def compare_structures(
    struct_1: StructureData,
    struct_2: StructureData
):
    """Determine if two structures are equivalent.
    If so, return True, else False.
    """
    struct_1, struct_2 = struct_1.get_pymatgen_structure(), struct_2.get_pymatgen_structure()
    sm = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=7)  # a bit looser than default
    try:
        max_dist = sm.get_rms_dist(struct_1, struct_2)[1]
        if max_dist < 0.1:
            return True
        else:
            return False
    except:
        return False # couldnt match structures, so not the same


def get_options_dict(
    computer: str,
):
    """
    Get the options for a given computer.
    """
    options = loadfn(os.path.join(MODULE_DIR, "yaml_files/options.yaml"))
    if not computer in options.keys():
        raise ValueError(
            f"Computer {computer} not found in options.yaml! "
            +"Please update this file"
        )
    return options[computer]


def setup_options(
    hpc_string : str,
    num_machines: int,
    num_mpiprocs_per_machine: int=128,  # assume archer2
    num_cores_per_machine: int=128,  # assume archer2
    time_in_hours: int=24,
    priority: bool=True,
    account: str=None,
):
    """Setup HPC options for the workchain.
    This depends on the hpc chosen by user.
    """
    options = get_options_dict(hpc_string)
    if "archer" in hpc_string.lower():
        options.update({
            'resources':
                {
                    'num_machines': num_machines,
                    'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
                    'num_cores_per_machine': num_cores_per_machine,
                },
            'max_wallclock_seconds': int(time_in_hours*3600),
        })
        options.update({
            "qos": "standard" if priority else "lowpriority"
        })
        # Update account if specified
        if account in [
            "e05-pool", "e05-gc-wal", "e05-discov-wal", "e05-free", "e05-low"
        ]:
            options["account"] = account
    elif "young" in hpc_string.lower():
        options.update({
            'resources':
                {
                    "tot_num_mpiprocs": int(num_machines * num_mpiprocs_per_machine),
                    "parallel_env": "mpi"
                },
            'max_wallclock_seconds': int(time_in_hours*3600),
        })
        # Update Queue priority
        options.update({
            'account': 'Gold' if priority else 'Free'
        })
    return options


def setup_settings(calc_type: str) -> dict:
    """Setup settings for workchain.
    This specifies which outputs to parse."""
    if calc_type == "snb":
        return {
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
                'add_forces' : True,
                "add_stress": True,
                "add_energies": True,
                'add_trajectory': True,
                # Dont parse:
                "add_dos": False,
                'add_kpoints': False,
                'add_bands': False,
            },
        }
    elif calc_type == "screening":
        return {
            "parser_settings": {
                "misc": [
                    "total_energies",
                    "maximum_force",
                    "maximum_stress",
                    "run_status",
                    "run_stats",
                    "notifications",
                ],
                "add_structure": True,  # retrieve structure
                # Dont parse:
                'add_forces' : False,
                "add_stress": False,
                "add_energies": False,
                'add_trajectory': False,
                "add_dos": False,
                'add_kpoints': False,
                'add_bands': False,
            },
        }
    elif calc_type == "relax_bulk":
        return {
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
                "add_stress": False,
                "add_forces": False,
                'add_trajectory': False,
                "add_dos": False,
                'add_kpoints': False,
                'add_bands': False,
            },
        }
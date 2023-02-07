"""Query material from materials project database, and submit relaxation"""

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
    k_density: Float,
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


def compare_structures(struct_1, struct_2):
    """Determine if two structures are equivalent."""
    sm = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=7)  # a bit looser than default
    try:
        max_dist = sm.get_rms_dist(struct_1, struct_2)[1]
        if max_dist < 0.1:
            return True
        else:
            return False
    except:
        return False # couldnt match structures, so not the same
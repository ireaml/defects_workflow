"""Query material from materials project database, and submit relaxation"""

from copy import deepcopy

# pymatgen
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core.structure import Structure

# aiida
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.orm import Dict, Float, Str, StructureData, KpointsData, Code
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import submit, calcfunction

# Parameters:
api_key="MsKnfQSzWAraK6zyhZ7OlNTlVl2GMuWr"

# Query material from materials project database based on mp-id
@calcfunction
def query_mp(
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


# Calculate the kpoints required for the relaxation based on density
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
    kpoints_data = KpointsData().set_kmesh(kpoints.kpts)
    return kpoints_data



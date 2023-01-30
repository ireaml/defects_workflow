"""
Collection of useful functions for vasp input
"""

import math
import os
import warnings
from typing import Optional
import numpy as np
import yaml

from monty.io import zopen
from monty.serialization import dumpfn, loadfn

import pymatgen.core.structure
from pymatgen.core.periodic_table import Element
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.inputs import Potcar

warnings.filterwarnings("ignore")  # ignore potcar warnings

# In-house stuff
from vasp_toolkit.potcar import get_potcar_mapping, get_number_of_electrons

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(
    os.path.join(MODULE_DIR, "../yaml_files/vasp/default_POTCARs.yaml")
)


def get_default_number_of_bands(
    structure: pymatgen.core.structure.Structure,
    number_of_electrons: Optional[int] = None,
    potcar_mapping: Optional[dict] = None,
) -> int:
    """
    Returns the default number of bands that will be used by vasp, i.e. NELECT/2+NIONS/2 for non-spinpolarized \
        (without considering paralelization restrictions)
    Args:
        structure (pymatgen.core.structure.Structure):
            pymatgen Structure object
        number_of_electrons (int):
            number of electrons in your system
        potcar_mapping (dict):
            dict mapping element symbols to POTCARs symbols
    Returns:
        int: number of bands that will be used by VASP by default
    """
    if not potcar_mapping:
        # Use default potcar mapping (recommened ones by VASP)
        potcar_mapping = get_potcar_mapping(structure)
    if not number_of_electrons:
        number_of_electrons = get_number_of_electrons(
            structure=structure, potcar_mapping=potcar_mapping
        )
    number_of_ions = len(structure)
    return max(
        math.ceil(number_of_electrons / 2 + number_of_ions / 2),  # number of bands
        math.ceil(number_of_electrons * 0.6),
    )


def get_number_of_bands(
    kpar: int,
    nbands_required: int,
    ncore: int = 14,
    nodes: int = 6,
    cores_per_node: int = 28,
    verbose: bool = True,
) -> int:
    """
    Given the VASP parallelization settings (KPAR, NCORE), number of cores
    and minimum number of bands, it returns the rounded number of bands that will
    be used by VASP.
    """
    cores = nodes * cores_per_node
    cores_per_kpoint = cores / kpar
    print("Cores per kpoint: ", cores_per_kpoint)
    npar = cores_per_kpoint / ncore
    print("NPAR: ", npar)

    # Now set NBANDS multiple of npar
    nbands_divided_by_npar = nbands_required / npar
    nbands_optimum = math.ceil(nbands_divided_by_npar) * npar
    if verbose:
        print(f"Minimum number of bands you require: {nbands_required}")
        print(
            "NBANDS used by VASP (applying parallelization requirements): ",
            nbands_optimum,
        )
    return nbands_optimum


def check_paralellization(
    kpar: int,
    nodes: int,
    cores_per_node: int,
    ncore: int,
    number_of_bands: Optional[int] = None,
    structure: Optional[pymatgen.core.structure.Structure] = None,
    potcar_mapping: Optional[dict] = None,
) -> None:
    """
    Check parallelization settings for VASP calculation-

    Args:
        kpar (int):
            KPAR tag in VASP input file
        nodes (int):
            Total number of nodes
        cores_per_node (int):
            Number of cores per node
        ncore (int):
            NCORE tag in VASP input file
        number_of_bands (int):
            Number of bands set by user.
            If not set, it will be set to VASP default number of bands
    """
    if not number_of_bands:
        number_of_bands = get_default_number_of_bands(
            structure=structure,
            potcar_mapping=potcar_mapping,
        )  # Default number of bands that VASP would select for this structure & potcars
    number_of_bands_used = get_number_of_bands(
        kpar=kpar,
        nbands_required=number_of_bands,
        ncore=ncore,
        cores_per_node=cores_per_node,
        nodes=nodes,
        verbose=False,
    )
    cores = nodes * cores_per_node
    print("Total number of cores: ", cores)
    assert cores % kpar == 0, "Number of cores not divisible by KPAR!"
    print("Cores per kpoint:", cores / kpar)
    npar = cores / (kpar * ncore)
    print("NPAR:", npar)
    print(
        "Default number of bands that you/VASP would select for system:",
        number_of_bands,
    )
    print("Rounded number of bands (multiple of NPAR): ", number_of_bands_used)

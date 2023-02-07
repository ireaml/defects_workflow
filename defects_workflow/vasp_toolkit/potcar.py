"""
Collection of useful functions to setup/manipulate VASP potcars
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
from pymatgen.io.vasp.inputs import Potcar, UnknownPotcarWarning

warnings.filterwarnings(
    "ignore", category=UnknownPotcarWarning
)  # Ignore pymatgen POTCAR warnings

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_potcar_dict = loadfn(
    os.path.join(MODULE_DIR, "../yaml_files/vasp/default_POTCARs.yaml")
)


def get_potcar_mapping(
    structure: pymatgen.core.structure.Structure,
) -> dict:
    """
    Create dictionary matching element symbol (as str) to vasp potcar name (as str)
    for the elements present in the input structure.

    Args:
        structure (pymatgen.core.structure.Structure):
            pymatgen structure object

    Returns:
        dict: element symbols matching the potcar symbols
    """
    potcar_mapping = {
        str(element): default_potcar_dict["POTCAR"][str(element)]
        for element in structure.composition.elements
    }
    return potcar_mapping


def get_potcar_valence_electrons(
    potcar_mapping: dict,
) -> dict:
    """
    Makes a dict matching element symbol to the number of valence electrons
    for selected pseudopotentials.
    Args:
        potcar_mapping (dict):
    Returns:
        dict: matching element symbols to the number of valence electrons in the
        corresponding pseudopotential
    """
    symbols = list(potcar_mapping.values())
    potcar = Potcar(symbols=symbols)
    potcar_dict = dict(zip(potcar_mapping.keys(), potcar))

    return {key: potcarsingle.nelectrons for key, potcarsingle in potcar_dict.items()}


def get_potcar_from_structure(
    structure: pymatgen.core.structure.Structure,
    potcar_mapping: Optional[dict] = None,
) -> Potcar:
    """
    Returns Potcar object for the elements present in your structure and
    the vasp potcars selected in the dictionary potcar_mapping.

    Args:
        structure (pymatgen.core.structure.Structure):
            pymatgen Structure object
        potcar_mapping (dict):
            dict matching element string to potcar symbol
            (e.g. {'Pb': 'Pb_d'})

    Returns:
        pymatgen-io.vasp.inputs.Potcar: pymatgen Potcar object
    """
    # if potcar_mapping is specified, check elements in structure are the same than in potcar_mapping
    if potcar_mapping:
        assert set(structure.composition.elements) == {
            Element(element_string) for element_string in potcar_mapping
        }
    else:  # if not given, use default potcars
        potcar_mapping = get_potcar_mapping(structure)

    # Check the order of element symbols is the same in POTCAR and structure
    ordered_symbols = [
        potcar_mapping[str(element)] for element in structure.composition.elements
    ]

    return Potcar(symbols=ordered_symbols)


def get_potcars_from_mapping(potcar_mapping: dict) -> Potcar:
    """
    Returns POTCARS for the elements and POTCAR symbols given in potential mapping.
    Args:
        potcar_mapping (dict):
            matches element symbol to VASP POTCAR name, i.e {'Pb': 'Pb_d', }
    """
    symbols = list(potcar_mapping.values())
    potcars = Potcar(symbols=symbols)
    for potcar_single in potcars:
        print(potcar_single.element)
        print(potcar_single.electron_configuration)
    return potcars


def get_number_of_electrons(
    structure: pymatgen.core.structure.Structure,
    potcar_mapping: Optional[dict] = None,
) -> int:
    """
    Calculate the number of electrons for structure with chosen pseudopotentials
    (specicied in potential_mapping dict)

    Args:
        structure (pymatgen.core.structure.Structure):
            pymatgen structure object
        potcar_mapping (dict):
            dict matching element symbol (str) to pseudopotential
            name (i.e: {'Sn': 'Sn_d'})

    Returns:
        int: number of electrons
    """
    if not potcar_mapping:
        potcar_mapping = get_potcar_mapping(structure)
    potcar_valence_electrons = get_potcar_valence_electrons(potcar_mapping)
    nelect = 0
    for element in structure.composition:
        nelect += (
            structure.composition[element] * potcar_valence_electrons[element.symbol]
        )
    return nelect


def get_valence_orbitals_from_potcar(potcar: Potcar) -> dict:
    """Get valence orbitals (in the form of a dictionary mapping
    element to its valence orbitals) from pymatgen Potcar object.
    """
    orbitals = {}
    for potcarsingle in potcar:
        orbitals[potcarsingle.element] = [
            orbital[1] for orbital in potcarsingle.electron_configuration
        ]
    return orbitals

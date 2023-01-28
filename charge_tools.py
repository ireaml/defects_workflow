"""
Reproduced from https://github.com/alexsquires/defectivator/blob/main/defectivator/tools.py
"""

import numpy as np
import pandas as pd
from defectivator.interstitials import InterstitialMap
from copy import deepcopy
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

data = get_data('charges.csv')

def extend_list_to_zero(l: list) -> np.array:
    """
    find the nearest integer to 0 in the list, and include all numbers between
    the nearest integer and 0 in the list.
    args:
        l (list): list of numbers
    returns:
        np.array: array of numbers, extended to 0
    """
    furthest_int = max([i for i in l if i != 0], key=lambda x: abs(x))
    if furthest_int < 0:
        return np.arange(furthest_int, 0 + 1, 1)
    else:
        return np.arange(0, furthest_int + 1, 1)


def get_charges(atom: str, charge_tol: float = 5) -> np.array:
    """
    For a given atom, parse the oxidation states of that atom present in the
    ICSD and return a range of charges from the highest to lowest possible
    oxidation states. The charges are only considered if they represent greater
    than `charge_tol`% of all the oxidation states of the atoms in the ICSD
    ICSD data taken from: `doi.org/10.1021/acs.jpclett.0c02072`
    args:
        atom (str): atom to get oxidation states for
        charge_tol (float): tolerance for oxidation states in percentage
    returns:
        np.array: array of reasonable oxidation states for atom
    """
    ox_states = pd.read_csv(data, delim_whitespace=True, index_col="Element")
    perc = ox_states.loc[atom] / ox_states.loc[atom].sum() * 100
    charges = [int(k) for k, v in perc.items() if v > charge_tol]
    if 0 not in charges:
        charges.append(0)
    return np.arange(min(charges), max(charges) + 1, 1, dtype=int)


def charge_identity(atom: str, charge_tol: float) -> str:
    """
    Define whether we consider an atom to be an anion, cation or amphoteric
    args:
        atom (str): atom to check
        charge_tol (float): tolerance for oxidation states in percentage
    returns:
        str: "anion", "cation" or "amphoteric"
    """
    charges = get_charges(atom, charge_tol)
    if all(charges >= 0):
        return "cation"
    elif all(charges <= 0):
        return "anion"
    else:
        return "both"


    def _get_antisite_charges(self, site, sub) -> list[int]:
        site_charges = get_charges(site, self.charge_tol) * -1
        sub_charges = get_charges(sub, self.charge_tol)

        if sub in self.cations and site in self.cations:
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
        elif sub in self.anions and site in self.anions:
            charges = np.arange(
                min(sub_charges) + min(site_charges), min(sub_charges) + 1, 1, dtype=int
            )
            if 0 not in charges:
                charges = extend_list_to_zero(charges)
            return sorted(charges)


def group_ions(species, atom_type, charge_tol=5):
    """Given a list of atoms, find all the cations
    or anions depending on the atom_type arg
    args:
        species (list[str]): list of atoms to pick out all the anions or cations
        atoms_type (str): whether to search for ions classed as "anion" or "cation"
        charge_tol (float): charge_tolerance to pass to charge_indentity
    returns:
        ions (list[str]): list of atoms which have been defined as either anions
            or cations.
    """
    ions = [
        a
        for a in species
        if charge_identity(a, charge_tol) == atom_type
        or charge_identity(a, charge_tol) == "both"
    ]
    return ions
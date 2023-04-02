"""Code to generate neutral vacancy defects.
"""

import os
from copy import deepcopy

from pymatgen.core.structure import Structure

from shakenbreak.input import Distortions

from defect_generation import _generate_defects

class SnBVacancy():

    def __init__(
        self,
        structure: Structure,
        defect_types=["vacancies"],
        only_neutral_defects=True,
        charge_tolerance=13,
        supercell_max_number_atoms=150,
        supercell_min_number_atoms=45,
        supercell_min_length=10.0,
    ):
        """Generate neutral vacancy defects.

        Args:
            structure (Structure): Input structure.
            defect_types (list): List of defect types to generate.
            only_neutral_defects (bool): Whether to only generate neutral defects.
            supercell_max_number_atoms (int): Maximum number of atoms in the supercell.
            supercell_min_length (float): Minimum length of the supercell.
        """
        self.composition = structure.composition.reduced_formula
        self.structure = structure
        self.defect_types = defect_types
        self.only_neutral_defects = only_neutral_defects
        self.charge_tolerance = charge_tolerance
        self.supercell_max_number_atoms = supercell_max_number_atoms
        self.supercell_min_number_atoms = supercell_min_number_atoms
        self.supercell_min_length = supercell_min_length

    def generate_defects(self):
        """Generate defects.

        Returns:
            defects (list): List of defects.
        """
        self.defects = _generate_defects(
            bulk=self.structure,
            defect_types=self.defect_types,
            min_atoms=self.supercell_min_number_atoms,
            max_atoms=self.supercell_max_number_atoms,
            min_length=self.supercell_min_length,
            force_diagonal=False,
            dummy_species_str="X",
            only_neutral_defects=self.only_neutral_defects,
            charge_tolerance=self.charge_tolerance,
        )

        return self.defects

    def get_cation_vacancies(self):
        """Get cation vacancies.

        Returns:
            cation_vacancies (list): List of cation vacancies.
        """
        if self.defects is None:
            self.generate_defects()

        # Only select vacancies
        vacancies = self.defects['vacancies']
        self.cation_vacancies = deepcopy(vacancies)

        # Only cation vacancies
        for defect_name, defect_entry_list in vacancies.items():
            d = defect_entry_list[0].defect  # first charged defect
            site_index = d.defect_site_index
            oxi_state = d.structure[site_index].specie.oxi_state
            if oxi_state < 0:
                print("Removing vacancy", defect_name)
                self.cation_vacancies.pop(defect_name)

        return self.cation_vacancies

    def apply_snb(
        self,
        defects: dict,
        ncore: int = 10,
        encut: int = 350,
    ):
        """Apply ShakeNBreak to defects"""
        # Create directory for composition
        if not os.path.exists(self.composition):
            os.mkdir(self.composition)
        dist = Distortions(defects=defects,)
        distorted_dict, metadata_dict = dist.write_vasp_files(
            incar_settings={
                "NCORE": ncore,
                "ENCUT": encut,
            },
            output_path=self.composition,
        )
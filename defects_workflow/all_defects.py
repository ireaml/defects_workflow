"""Code to generate neutral defects & apply SnB to them.
"""

import os
from copy import deepcopy

from pymatgen.core.structure import Structure

from shakenbreak.input import Distortions

from defects_workflow.defect_generation import _generate_defects

class SnBDefects():

    def __init__(
        self,
        structure: Structure,
        defect_types=["vacancies", "antisites", "interstitials"],
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
        self.defects = None

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

    def format_defects(self):
        """Get cation vacancies.

        Returns:
            cation_vacancies (list): List of cation vacancies.
        """
        if not self.defects:
            self.generate_defects()

        # Only select vacancies
        vacancies = self.defects['vacancies'].values()

        return vacancies

    def apply_shakenbreak(
        self,
        defects: dict,
        ncore: int = 10,
        encut: int = 350,
        output_path: str = None,
    ):
        """Apply ShakeNBreak to defects"""
        # Create directory for composition
        if output_path:
            if not self.composition in output_path:
                output_path = os.path.join(output_path, self.composition)
        else:
            output_path = self.composition
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        dist = Distortions(defects=defects,)
        distorted_dict, metadata_dict = dist.write_vasp_files(
            incar_settings={
                "NCORE": ncore,
                "ENCUT": encut,
            },
            output_path=output_path,
        )
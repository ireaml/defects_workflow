from __future__ import annotations

import numpy as np

from pymatgen.core import PeriodicSite, Structure
from pymatgen.analysis.defects.core import Defect, Vacancy, Substitution, Interstitial


class CustomDefect(Defect):
    """Modified Defect() class to enable storing
    supercell structure in the .supercell_structure attribute.
    """
    def __init__(
        self,
        structure: Structure,
        site: PeriodicSite,
        multiplicity: int | None = None,
        oxi_state: float | None = None,
        symprec: float = 0.01,
        angle_tolerance: float = 5,
        user_charges: list[int] | None = None,
    ) -> None:
        """Initialize a defect object.
        Args:
            structure: The structure of the defect.
            site: The site
            multiplicity: The multiplicity of the defect.
            oxi_state: The oxidation state of the defect, if not specified,
                this will be determined automatically.
            symprec: Tolerance for symmetry finding.
            angle_tolerance: Angle tolerance for symmetry finding.
            user_charges: User specified charge states. If specified,
                ``get_charge_states`` will return this list. If ``None`` or empty list
                the charge states will be determined automatically.
        """
        self.structure = structure
        self.site = site
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.multiplicity = (
            multiplicity if multiplicity is not None else self.get_multiplicity()
        )
        self.user_charges = user_charges if user_charges else []
        if oxi_state is None:
            # TODO this step might take time so wrap it in a timer
            self.structure.add_oxidation_state_by_guess()
            self.oxi_state = self._guess_oxi_state()
        else:
            self.oxi_state = oxi_state
        self.supercell_structure = None

    @classmethod
    def from_defect(cls, defect):
        """Initialize a CustomDefect object from a Defect object."""
        return cls(
            structure=defect.structure,
            site=defect.site,
            multiplicity=defect.multiplicity,
            oxi_state=defect.oxi_state,
            symprec=defect.symprec,
            angle_tolerance=defect.angle_tolerance,
            user_charges=defect.user_charges,
        )

    def get_supercell_structure(
        self,
        sc_mat: np.ndarray | None = None,
        dummy_species: str | None = None,
        min_atoms: int = 80,
        max_atoms: int = 240,
        min_length: float = 10.0,
        force_diagonal: bool = False,
    ) -> Structure:
        """Generate the supercell for a defect.
        Args:
            sc_mat: supercell matrix if None, the supercell will be determined by `CubicSupercellAnalyzer`.
            dummy_species: Dummy species to highlight the defect position (for visualizing vacancies).
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal transformation matrix.
        Returns:
            Structure: The supercell structure.
        """
        self.supercell_structure = super().get_supercell_structure(
            sc_mat=sc_mat,
            dummy_species=dummy_species,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
            force_diagonal=force_diagonal
        )
        return self.supercell_structure


class CustomVacancy(Vacancy):
    """Modified Vacancy() class to enable storing
    supercell structure in the .supercell_structure attribute.
    """
    @classmethod
    def from_vacancy(cls, defect):
        """Initialize a CustomVacancy object from a Vacancy object."""
        return cls(
            structure=defect.structure,
            site=defect.site,
            multiplicity=defect.multiplicity,
            oxi_state=defect.oxi_state,
            symprec=defect.symprec,
            angle_tolerance=defect.angle_tolerance,
            user_charges=defect.user_charges,
        )

    def get_supercell_structure(
        self,
        sc_mat: np.ndarray | None = None,
        dummy_species: str | None = None,
        min_atoms: int = 80,
        max_atoms: int = 240,
        min_length: float = 10.0,
        force_diagonal: bool = False,
    ) -> Structure:
        """Generate the supercell for a defect.
        Args:
            sc_mat: supercell matrix if None, the supercell will be determined by `CubicSupercellAnalyzer`.
            dummy_species: Dummy species to highlight the defect position (for visualizing vacancies).
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal transformation matrix.
        Returns:
            Structure: The supercell structure.
        """
        self.supercell_structure = super(Vacancy, self).get_supercell_structure(
            sc_mat=sc_mat,
            dummy_species=dummy_species,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
            force_diagonal=force_diagonal
        )
        return self.supercell_structure

class CustomSubstitution(Substitution):
    """Modified Substitution() class to enable storing
    supercell structure in the .supercell_structure attribute.
    """
    def __init__(
        self,
        structure: Structure,
        site: PeriodicSite,
        multiplicity: int | None = None,
        oxi_state: float | None = None,
        **kwargs,
    ) -> None:
        """Initialize a substitutional defect object.
        The position of `site` determines the atom to be removed and the species of
        `site` determines the replacing species.
        Args:
            structure: The structure of the defect.
            site: Replace the nearest site with this one.
            multiplicity: The multiplicity of the defect.
            oxi_state: The oxidation state of the defect, if not specified,
            this will be determined automatically.
        """
        super().__init__(structure, site, multiplicity, oxi_state, **kwargs)
        self.supercell_structure = None

    @classmethod
    def from_substitution(cls, defect):
        """Initialize a CustomSubstitution object from a Substitution object."""
        return cls(
            structure=defect.structure,
            site=defect.site,
            multiplicity=defect.multiplicity,
            oxi_state=defect.oxi_state,
            symprec=defect.symprec,
            angle_tolerance=defect.angle_tolerance,
            user_charges=defect.user_charges,
        )

    def get_supercell_structure(
        self,
        sc_mat: np.ndarray | None = None,
        dummy_species: str | None = None,
        min_atoms: int = 80,
        max_atoms: int = 240,
        min_length: float = 10.0,
        force_diagonal: bool = False,
    ) -> Structure:
        """Generate the supercell for a defect.
        Args:
            sc_mat: supercell matrix if None, the supercell will be determined by `CubicSupercellAnalyzer`.
            dummy_species: Dummy species to highlight the defect position (for visualizing vacancies).
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal transformation matrix.
        Returns:
            Structure: The supercell structure.
        """
        self.supercell_structure = super(Substitution, self).get_supercell_structure(
            sc_mat=sc_mat,
            dummy_species=dummy_species,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
            force_diagonal=force_diagonal
        )
        return self.supercell_structure


class CustomInterstitial(Interstitial):
    """Modified Interstitial() class to enable storing
    supercell structure in the .supercell_structure attribute.
    """
    def __init__(
        self,
        structure: Structure,
        site: PeriodicSite,
        multiplicity: int = 1,
        oxi_state: float | None = None,
        **kwargs,
    ) -> None:
        """Initialize an interstitial defect object.
        The interstitial defect effectively inserts the `site` object into the structure.
        Args:
            structure: The structure of the defect.
            site: Inserted site, also determines the species.
            multiplicity: The multiplicity of the defect.
            oxi_state: The oxidation state of the defect, if not specified,
                this will be determined automatically.
        """
        super().__init__(structure, site, multiplicity, oxi_state, **kwargs)
        self.supercell_structure = None

    @classmethod
    def from_interstitial(cls, defect):
        """Initialize a CustomInterstitial object from an Interstitial object."""
        return cls(
            structure=defect.structure,
            site=defect.site,
            multiplicity=defect.multiplicity,
            oxi_state=defect.oxi_state,
            symprec=defect.symprec,
            angle_tolerance=defect.angle_tolerance,
            user_charges=defect.user_charges,
        )

    def get_supercell_structure(
        self,
        sc_mat: np.ndarray | None = None,
        dummy_species: str | None = None,
        min_atoms: int = 80,
        max_atoms: int = 240,
        min_length: float = 10.0,
        force_diagonal: bool = False,
    ) -> Structure:
        """Generate the supercell for a defect.
        Args:
            sc_mat: supercell matrix if None, the supercell will be determined by `CubicSupercellAnalyzer`.
            dummy_species: Dummy species to highlight the defect position (for visualizing vacancies).
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal transformation matrix.
        Returns:
            Structure: The supercell structure.
        """
        self.supercell_structure = super(Interstitial, self).get_supercell_structure(
            sc_mat=sc_mat,
            dummy_species=dummy_species,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            min_length=min_length,
            force_diagonal=force_diagonal
        )
        return self.supercell_structure
""""Workflow to apply SnB to intrinsic defects in a host structure:
1. Generate defects
2. Screen interstitials or defects in sym. ineq. sites
3. Apply SnB
4. Submit relaxations
"""
import os
from abc import ABCMeta
from copy import deepcopy
from monty.serialization import loadfn

from aiida.engine import WorkChain, ToContext, calcfunction, if_
from aiida import orm
from aiida.tools.groups import GroupPath

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import  VaspInputSet
from pymatgen.analysis.defects.thermo import DefectEntry

from shakenbreak.input import Distortions

from defects_workflow.utils import (
    setup_options,
    setup_settings,
    setup_gamma_kpoints,
    get_options_dict,
    compare_structures
)
from defects_workflow.relaxation import setup_relax_inputs
from defects_workflow.defect_generation import (
    generate_defects,
    sort_interstitials_for_screening
)
from defects_workflow.vasp_input import setup_incar_snb


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
path = GroupPath()


class DefectGenerationWorkChain(WorkChain, metaclass=ABCMeta):
    """
    WorkChain to setup supercell and generate point defects
    for the primitive/conventional input structure.

    Defect charge states are determined based on common oxidation
    states for the element (controlled with the input parameter
    `charge_tolerance`, that works as a threshold).

    Args:
        structure: StructureData
        defect_types: List
        symmetry_tolerance: Float
        supercell_min_length: Float
        supercell_max_number_atoms: Int
        supercell_min_number_atoms: Int
        charge_tolerance: Float

    Returns:
        defect_entries_dict (Dict)
    """
    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.outline(
            cls.generate_defects,
        )

        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=False,
            help='The host structure (primitive or conventional).'
        )
        spec.input(
            "defect_types",
            valid_type=orm.List,
            required=False,
            default=orm.List(['vacancies', 'interstitials', 'antisites']),
            help=("List of defect types to generate. "
                "E.g: orm.List(['vacancies', 'interstitials', 'antisites'])"
            )
        )
        spec.input(
            'symmetry_tolerance',
            valid_type=orm.Float,
            required=False,
            default=orm.Float(0.01),
            help='Symmetry tolerance for space group analysis on the input structure.',
        )
        spec.input(
            'charge_tolerance',
            valid_type=orm.Float,
            required=False,
            default=orm.Float(13),
            help=(
                "Tolerance for determining charge states of defects. "
                "The charges are only considered if they represent greater "
                "than `charge_tolerance`% of all the oxidation states of the atoms in the ICSD "
                "data taken from: `doi.org/10.1021/acs.jpclett.0c02072`."
                "(e.g. he smaller the value, the more charge states will be considered.) "
                "It has been tested and values within 5-13% are reasonable."
            ),
        )
        spec.input(
            "supercell_min_length",
            valid_type=orm.Float,
            required=False,
            default=orm.Float(10.0),
            help="The minimum length of the supercell."
        )
        spec.input(
            "supercell_max_number_atoms",
            valid_type=orm.Int,
            required=False,
            default=orm.Int(200),
            help="The maximum number of atoms when generating supercell."
        )
        spec.input(
            "supercell_min_number_atoms",
            valid_type=orm.Int,
            required=False,
            default=orm.Int(30),
            help="The minimum number of atoms when generating supercell."
        )
        spec.output(
            "defect_entries_dict",
            valid_type=orm.Dict,
            required=False,
        )

    def generate_defects(self):
        """Generate defects."""

        self.report(
            "Generating defects, adding charge states & setting up supercells."
        )

        # Generate defects with pymatgen-analysis-defects
        defects_dict_aiida = generate_defects(
            bulk=self.inputs.structure,
            defect_types=self.inputs.defect_types,
            min_length=self.inputs.supercell_min_length,
            symprec=self.inputs.symmetry_tolerance,
            min_atoms=self.inputs.supercell_min_number_atoms,
            max_atoms=self.inputs.supercell_max_number_atoms,
            force_diagonal=orm.Bool(False),
            interstitial_min_dist=orm.Float(1.0),
            dummy_species_str=orm.Str("X"),  # to keep track of frac coords in sc
            charge_tolerance=self.inputs.charge_tolerance,
        )  # type orm.Dict (not dict!)
        # In defects_dict_aiida, DefectEntries are stored as `dictionaries`,
        # should refactor to DefectEntrys before applying SnB

        self.out("defect_entries_dict", defects_dict_aiida)


class InterstitialScreeningWorkChain(WorkChain, metaclass=ABCMeta):
    """Workchain to screen interstitials"""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.outline(
            cls.setup,
            cls.screen_interstitials,
            cls.analyse_screening_results
        )

        # Inputs
        spec.input(
            'defects_dict_aiida',
            valid_type=orm.Dict,
            required=False,
            help=(
                "Dictionary with generated defects (as DefectEntries objects), formatted as "
                "the output of DefectGenerationWorkChain (e.g. "
                " {'vacancies': {'v_Cd_s0': [DefectEntry_as_dict, ...]}}"
                ")."
            )
        )
        spec.input(
            'code_string_vasp_gam',
            valid_type=orm.Str,
            required=False,
            default=orm.Str("vasp_gam_6.3.0"), # default is archer2
            help='Code string for vasp_gam',
        )
        spec.input(
            "num_nodes",
            valid_type=orm.Int,
            required=False,
            default=orm.Int(1),
            help=("Number of nodes to use for relaxations. "
            "Recommended values: 1 using archer2, 2 if using Young "
            "(e.g. to get aorund 60-100 cores).")
        )
        # Outputs:
        spec.output(
            "defects_dict_aiida",
            valid_type=orm.Dict,
            required=False,
            help=(
                "Dictionary with screened defects (as DefectEntries objects), "
                "formatted as the output of DefectGenerationWorkChain (e.g. "
                " {'vacancies': {'v_Cd_s0': [DefectEntry_as_dict, ...]}}"
                ")."
            )
        )
        # Exit codes:
        spec.exit_code(
            0,
            'NO_ERROR',
            message='the sun is shining',
        )
        spec.exit_code(
            501,
            'ERROR_PARSING_INPUT_DICT',
            message=("Problem parsing the input dictionary with the defect entries. "
            "Verify it's of the correct format.")
        )
        spec.exit_code(
            502,
            'ERROR_PARSING_BULK_STRUCTURE',
            message=("Problem parsing bulk pymatgen Structure.")
        )
        spec.exit_code(
            503,
            "ERROR_PARSING_INTERSTITIAL_STRUCTURE",
            message="Problem parsing interstitial structure."
        )
        spec.exit_code(
            400,
            'ERROR_SUB_PROCESS_FAILED',
            message="The `{cls}` workchain failed with exit status {exit_status}."
        )
        spec.exit_code(
            600,
            'ERROR_OUTPUT_STRUCTURE_NOT_FOUND',
            message="Couldnt parse output structure from workchain {pk}."
        )
        spec.exit_code(
            501,
            'ERROR_PARSING_OUTPUT',
            message="Couldnt parse output from workchain {pk}."
        )

    def setup(self):
        """Setup primitive/conventional input structure"""
        defects_dict = self.inputs.defects_dict_aiida.get_dict()
        try:
            # Grab a defect entry
            defect_entry_as_dict = list(defects_dict["interstitials"].values())[0][0]
        except:
            return self.exit_codes.ERROR_PARSING_INPUT_DICT

        try:
            pmg_structure = Structure.from_dict(defect_entry_as_dict["defect"]["structure"])
            self.ctx.structure = orm.StructureData(
                pymatgen_structure=pmg_structure
            )
        except:
            return self.exit_codes.ERROR_PARSING_BULK_STRUCTURE

    def screen_interstitials(self):
        """
        Screen interstitials. For each element, select the interstitials lower
        in energy (if difference > 1eV).
        Also used to identify if any initial interstitial structures
        relaxed to the same final one (in this case, only one is selected).

        The screening is performed by performing Gamma point relaxations
        (for a given element, select interstitials lower in energy).
        """
        self.report("Screening interstitials")

        # Get composition string
        self.ctx.composition = self.setup_composition(self.ctx.structure)

        # Get interstitials with sym ineq configurations (@calcfunction):
        self.ctx.int_dict_aiida = sort_interstitials_for_screening(
            self.inputs.defects_dict_aiida
        ).get_dict()  # as python dict, with DefectEntry_as_dicts!

        # Check if there are actually more than 1 configuration
        # for each type of interstitial:
        if not self.ctx.int_dict_aiida:
            self.report(
                "No interstitials to screen! "
                "There's only one configuration for each type of interstitial."
            )
            return self.exit_codes.NO_ERROR

        # Get incar (same for all sym ineq interstitials)
        try:
            structure = Structure.from_dict(
                list(
                    list(self.ctx.int_dict_aiida.values())[0].values()
                )[0]["sc_entry"]["structure"]
            )
        except:
            self.out("defects_dict_aiida", self.inputs.defects_dict_aiida)
            return self.exit_codes.ERROR_PARSING_INTERSTITIAL_STRUCTURE
        # Update ncore based on hpc of code:
        self.ctx.hpc_string = self._determine_hpc()
        self.ctx.ncore = self._get_ncore(hpc_string=self.ctx.hpc_string)
        # For screening, only relax neutral:
        incar_Dict = setup_incar_snb(  # calcfunction
            supercell=orm.StructureData(pymatgen_structure=structure),
            charge=orm.Int(0),
            incar_settings=orm.Dict(
                {"NCORE": self.ctx.ncore}
            ),
        )

        # Setup KpointsData
        self.ctx.gam_kpts = setup_gamma_kpoints()

        # Setup HPC options:
        self.ctx.number_cores_per_machine = self._setup_number_cores_per_machine(
            hpc_string=self.ctx.hpc_string
        )
        self.ctx.num_nodes = self.inputs.num_nodes
        self.ctx.options = self.setup_options(
            hpc_string=self.ctx.hpc_string,
            num_machines=self.ctx.num_nodes,
            num_cores_per_machine=self.ctx.number_cores_per_machine,
            num_mpiprocs_per_machine=self.ctx.number_cores_per_machine,
        )
        # Setup settings for screening
        settings = self.setup_settings(calc_type="screening")

        # Submit geometry optimisations for all interstitials at the same time:
        for general_int_name, defect_entry_dict in self.ctx.int_dict_aiida.items():
            for defect_name, defect_entry in defect_entry_dict.items():
                structure = Structure.from_dict(
                    defect_entry["sc_entry"]["structure"]
                )
                # Submit relaxation (gamma point):
                workchain, inputs = setup_relax_inputs(
                    code_string=self.inputs.code_string_vasp_gam.value,
                    # aiida config:
                    options=self.ctx.options,
                    settings=settings,
                    # VASP inputs:
                    structure_data=orm.StructureData(pymatgen_structure=structure),
                    kpoints_data=self.ctx.gam_kpts, # 1,1,1
                    incar_dict=deepcopy(incar_Dict.get_dict()),
                    use_default_incar_settings=False,
                    shape=False,
                    volume=False,
                    ionic_steps=600,
                    # Labels:
                    workchain_label=f"screen_{defect_name}",  # eg Te_i_s32
                )
                workchain = self.submit(workchain, **inputs)
                # Add to group
                self.add_workchain_to_group(
                    workchain=workchain,
                    group_label=f"defects_db/{self.ctx.composition}/02_interstitials_screening"
                )
                # To context:
                key = f"screen.{defect_name}"
                self.to_context(**{key: workchain})

    def analyse_screening_results(self):
        """
        Analyse screening results.
        If any initial interstitial structures relaxed to the same final one, select only one.
        If energy difference between two interstitials is > 1eV, select the one with the lowest energy.
        """
        self.report("Analyse interstitial screening results")

        # Loop over element speific interstitials: (Te_i, Cd_i)
        for defect_entry_dict in self.ctx.int_dict_aiida.values():
            energies = {}  # defect_name: energy of interstitial
            for defect_name in defect_entry_dict:
                # Validate workchains
                key = f"screen.{defect_name}"
                self.validate_finished_workchain(workchain_name=key)
                # Parse final energy
                energies[defect_name] = self.parse_final_energy(workchain=self.ctx[key])
            # Compare energies:
            lowest_energy = min(energies.values())
            for defect_name, energy in energies.items():
                if abs(lowest_energy) - abs(energy) > 1:
                    # too high in energy, remove from dict
                    self.inputs.defects_dict_aiida["interstitials"].pop(defect_name, None)
                elif [
                    abs(energy - stored_energy) < 0.1 for stored_energy in energies.values()
                ]:
                    # energy differece with other structure is too small, likely same structure
                    # Use StructureMatcher to check
                    # Get defect_names of structures with same energy:
                    other_defect_name = [
                        k for k
                        in energies
                        if (abs(energies[defect_name] - energy) < 0.1 and k != defect_name)
                    ]
                    # Get structures:
                    structure_1 = self.parse_relaxed_structure(
                        workchain=self.ctx[f"screen.{defect_name}"]
                    )
                    structure_2 = self.parse_relaxed_structure(
                        workchain=self.ctx[f"screen.{other_defect_name}"]
                    )
                    if compare_structures(structure_1, structure_2):
                        self.inputs.defects_dict_aiida["interstitials"].pop(defect_name, None)

        self.out("defects_dict_aiida", self.inputs.defects_dict_aiida)

    def setup_composition(self, structure_data: orm.StructureData):
        """Setup composition."""
        structure = structure_data.get_pymatgen_structure()
        composition = structure.composition.to_pretty_string()
        return composition

    def _determine_hpc(self):
        code = orm.load_code(self.inputs.code_string_vasp_gam.value)
        return code.computer.label

    def _setup_number_cores_per_machine(self, hpc_string: str):
        """Determine number of cores based on HPC"""
        if "archer" in hpc_string.lower():
            return 128
        elif "young" in hpc_string.lower():
            return 40
        else:
            raise ValueError("HPC string not recognised.")

    def _get_ncore(self, hpc_string: str):
        """Determine NCORE based on HPC chosen"""
        if "archer" in hpc_string.lower():
            return 8  # or 16
        elif "young" in hpc_string.lower():
            return 10

    def setup_options(
        self,
        hpc_string : str,
        num_machines: int,
        num_mpiprocs_per_machine: int=128,  # assume archer2
        num_cores_per_machine: int=128,  # assume archer2
        time_in_hours: int=12,
    ):
        """Setup options for HPC with aiida."""
        return setup_options(
            hpc_string=hpc_string,
            num_machines=num_machines,
            num_mpiprocs_per_machine=num_mpiprocs_per_machine,
            num_cores_per_machine=num_cores_per_machine,
            time_in_hours=time_in_hours
        )

    def setup_settings(self, calc_type: str="screening"):
        return setup_settings(calc_type)

    def parse_relaxed_structure(self, workchain) -> orm.StructureData:
        """Get output (relaxed) structure from workchain."""
        try:
            return workchain.outputs.relax.structure
        except:
            self.exit_codes.ERROR_OUTPUT_STRUCTURE_NOT_FOUND.format(
                pk=workchain.pk
            )

    def parse_final_energy(self, workchain) -> float:
        """Parse final energy from relaxation workchain."""
        try:
            return workchain.outputs.misc.get_dict()['total_energies']['energy_extrapolated']
        except:
            self.exit_codes.ERROR_PARSING_OUTPUT.format(
                pk=workchain.pk
            )

    def add_workchain_to_group(self, workchain, group_label):
        """Add workchain to group."""
        group_path = GroupPath()
        if group_label:
            group = group_path[group_label].get_or_create_group()
            group = group_path[group_label].get_group()
            group.add_nodes(workchain)
            # print(
            #     f"Submitted relax workchain with pk: {workchain.pk}"
            #     +f" and label {workchain.label}, "
            #     +f"stored in group with label {group_label}"
            # )
        # else:
        #     print(
        #         f"Submitted relax workchain with pk: {workchain.pk} and label {workchain.label}"
        #     )

    def validate_finished_workchain(self, workchain_name: str):
        """Validate that the workchain finished successfully."""
        if workchain_name not in self.ctx:
            raise RuntimeError(f"Workchain {workchain_name} not found in context.")

        workchain = self.ctx[workchain_name]
        cls = self._process_class.__name__
        # Check if workchain finished successfully:
        if not workchain.is_finished_ok:
            exit_status = self.ctx.workchain.exit_status
            self.report(
                f"{cls}<{self.workchain.pk}> (label={workchain.label}) failed with exit status {exit_status}."
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=cls,
                exit_status=exit_status
            )
        # All ok
        self.report(
            f'{cls}<{self.ctx.workchain.pk}> finished successfully.'
        )


class ShakeNBreakWorkChain(WorkChain, metaclass=ABCMeta):
    """WorkChain to apply ShakeNBreak, submit relaxations and
    analyse results"""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.outline(
            cls.setup,
            cls.apply_shakenbreak,
            if_(cls.should_submit_relax)(
                cls.relax_defects,
                cls.parse_snb_relaxations,
                cls.results,
            ),
        )

        spec.input(
            'defects_dict_aiida',
            valid_type=orm.Dict,
            required=False,
            help=(
                "Dictionary with generated defects (as DefectEntries objects), formatted as "
                "the output of DefectGenerationWorkChain (e.g. "
                " {'vacancies': {'v_Cd_s0': [DefectEntry_as_dict, ...]}}"
                ")."
            )
        )
        spec.input(
            'submit_relaxations',
            valid_type=orm.Bool,
            required=False,
            default=orm.Bool(True), # default is archer2
            help='Whether to submit relaxations of defects.',
        )
        # Code/HPC:
        spec.input(
            'code_string_vasp_gam',
            valid_type=orm.Str,
            required=False,
            default=orm.Str("vasp_gam_6.3.0"), # default is archer2
            help='Code string for vasp_gam',
        )
        spec.input(
            "num_nodes",
            valid_type=orm.Int,
            required=False,
            default=orm.Int(1),
            help=("Number of nodes to use for relaxations. "
            "Recommended values: 1 using archer2, 2 if using Young "
            "(e.g. to get aorund 60-100 cores).")
        )
        # Outputs:
        spec.output(
            'results',
            valid_type=orm.Dict,
            required=False,
        )
        spec.output(
            "snb_output_dicts",
            valid_type=orm.Dict,
            required=False,
        )
        # Exit codes:
        spec.exit_code(
            0,
            'NO_ERROR',
            message='the sun is shining',
        )
        spec.exit_code(
            501,
            'ERROR_PARSING_INPUT_DICT',
            message=("Problem parsing the input dictionary with the defect entries. "
            "Verify it's of the correct format.")
        )
        spec.exit_code(
            502,
            'ERROR_PARSING_BULK_STRUCTURE',
            message=("Problem parsing bulk pymatgen Structure.")
        )
        spec.exit_code(
            400,
            'ERROR_SUB_PROCESS_FAILED',
            message="The `{cls}` workchain failed with exit status {exit_status}."
        )
        spec.exit_code(
            600,
            'ERROR_OUTPUT_STRUCTURE_NOT_FOUND',
            message="Couldnt parse output structure from workchain {pk}."
        )
        spec.exit_code(
            501,
            'ERROR_PARSING_OUTPUT',
            message="Couldnt parse output from workchain {pk}."
        )

    def setup(self):
        """Setup primitive/conventional input structure"""
        defects_dict = self.inputs.defects_dict_aiida.get_dict()
        try:
            # Grab a defect entry
            defect_entry_as_dict = list(list(defects_dict.values())[0].values())[0][0]
        except:
            return self.exit_codes.ERROR_PARSING_INPUT_DICT

        try:
            pmg_structure = Structure.from_dict(defect_entry_as_dict["defect"]["structure"])
            self.ctx.structure = orm.StructureData(
                pymatgen_structure=pmg_structure
            )
        except:
            return self.exit_codes.ERROR_PARSING_BULK_STRUCTURE

    def should_submit_relax(self):
        """Whether to submit relaxations"""
        return self.inputs.submit_relaxations

    def apply_shakenbreak(self):
        """Apply ShakeNBreak."""
        self.report("Applying shakenbreak")

        output_Dict = apply_shakenbreak(  # calcfunction
            defects_Dict=self.inputs.defects_dict_aiida  # dict with all pmg objects as dicts
        )
        self.ctx.distorted_dict_aiida = output_Dict["distortions_dict"]  # this is a python dict
        self.out("snb_output_dicts", output_Dict)  # Both distortions & metadata dict
        # Note that in distortions_dict, all pmg objects are dicts!
        # This distortions_dict is formatted like:
        # {defect_name: {
        #    "defect_site": Site_as_dict,
        #    "defect_type": Site_as_dict,
        #    "defect_multiplicity: ,
        #    "defect_supercell_site": ,
        #    "charges": {
        #        charge: {
        #             "structures": {
        #                "Unperturbed": Structure_as_dict,
        #                "distortions": {"Bond_Distortion_-60.0%": Structure,}
        #             }
        #        }
        #    }
        # }

    def relax_defects(self):
        """Submit geometry optimisations for the defects."""
        self.report("relax_defects")
        # Get composition string
        self.ctx.composition = self.setup_composition(self.ctx.structure)
        # Setup settings
        settings = self.setup_settings(calc_type="snb")
        # Setup options, and vasp input
        self.ctx.hpc_string = self._determine_hpc()
        # Setup KpointsData
        self.ctx.gam_kpts = setup_gamma_kpoints()

        # Setup HPC options:
        self.ctx.number_cores_per_machine = self._setup_number_cores_per_machine(
            hpc_string=self.ctx.hpc_string
        )
        self.ctx.num_nodes = self.inputs.num_nodes
        self.ctx.ncore = self._get_ncore(hpc_string=self.ctx.hpc_string)
        self.ctx.options = self.setup_options(
            hpc_string=self.ctx.hpc_string,
            num_machines=self.ctx.num_nodes,
            num_cores_per_machine=self.ctx.number_cores_per_machine,
            num_mpiprocs_per_machine=self.ctx.number_cores_per_machine,
        )

        distorted_dict = self.ctx.distorted_dict_aiida # all pmg objects as dicts
        # Loop over defects, charge states & distortions:
        for defect_name, dist_dict in distorted_dict.items():
            for charge in dist_dict["charges"]:
                # Get incar for unperturbed (same for all distorted structures)
                structure = Structure.from_dict(
                    dist_dict["charges"][charge]["structures"]["Unperturbed"]
                )  # dict -> Structure
                incar_Dict = setup_incar_snb(  # calcfunction
                    supercell=orm.StructureData(pymatgen_structure=structure),
                    charge=orm.Int(charge),
                    incar_settings=orm.Dict(
                        {"NCORE": self.ctx.ncore}  # update ncore based on hpc of code
                    ),
                )
                # Submit unperturbed structure:

                # Submit each distortion:
                for dist_name, structure_as_dict in zip(
                    # Names:
                    [
                        "Unperturbed",
                    ]
                    + list(
                        dist_dict["charges"][charge]["structures"][
                            "distortions"
                        ].keys()
                    ),
                    # Structures (as dicts):
                    [dist_dict["charges"][charge]["structures"]["Unperturbed"]]
                    + list(
                        dist_dict["charges"][charge]["structures"][
                            "distortions"
                        ].values()
                    ),
                ):
                    # Submit relaxation (gamma point):
                    structure = Structure.from_dict(structure_as_dict)  # dict -> Structure
                    workchain, inputs = setup_relax_inputs(
                        code_string=self.inputs.code_string_vasp_gam.value,
                        # aiida config:
                        options=self.ctx.options,
                        settings=settings,
                        # VASP inputs:
                        structure_data=structure,
                        kpoints_data=self.ctx.gam_kpts, # 1,1,1
                        # Incar settings:
                        incar_dict=deepcopy(incar_Dict.get_dict()),
                        use_default_incar_settings=False,
                        shape=False,
                        volume=False,
                        ionic_steps=600,
                        # Labels:
                        workchain_label=f"snb.{defect_name}.{charge}.{dist_name}",  # e.g. snb.Te_i_s32.0.Unperturbed
                    )
                    workchain = self.submit(workchain, **inputs)
                    # Add to group
                    self.add_workchain_to_group(
                        workchain=workchain,
                        group_label=f"defects_db/{self.ctx.composition}/03_snb_relaxations"
                    )
                    key = f"snb.{defect_name}.{charge}.{dist_name}"
                    self.to_context(**{key: workchain})

    def parse_snb_relaxations(self):
        """
        Validate & parse data from the relaxations.
        The output dictionary is formatted like:
        {
            defect_name:
                "charges": {
                    charge_state: {
                        Unperturbed: {},
                        Bond_Distortion: {},
                        ...
                    }
                },
                "defect_entry_dict": {},
        }
        """
        self.report("parse_snb_relaxations")
        distorted_dict = self.ctx.distorted_dict_aiida  # all pmg objects as dicts
        out_dict = {}
        for defect_name, dist_dict in distorted_dict.items():
            out_dict[defect_name] = {
                "defect_entries": self.inputs.defects_dict_aiida[defect_name],  # list with DefectEntry_as_dict for all charge states
                "charges": {},
            }
            for charge in dist_dict["charges"]:
                out_dict[defect_name]["charges"][charge] = {}
                for dist_name in (
                    [
                        "Unperturbed",
                    ]
                    + list(
                        dist_dict["charges"][charge]["structures"]["distortions"].keys()
                    )
                ):
                    # Validate and parse outputs (structure, energy, traj, forces, stress)
                    key = f"snb.{defect_name}.{charge}.{dist_name}"
                    self.validate_finished_workchain(workchain_name=key)
                    out_dict[defect_name]["charges"][charge][dist_name] = parse_snb_workchain(
                        workchain=self.ctx[key]
                    ).get_dict()

        self.ctx.out_dict = orm.Dict(out_dict)

    def results(self):
        """Attach the remaining output results.
        The output dictionary has the following format:
        {defect_name: {charge: {dist_name: {"pk":, "structure": , etc}, }}}
        """
        if "out_dict" not in self.ctx:
            self.report("No results to attach. Workchain failed.")
        else:
            self.out("results", self.ctx.out_dict)
            self.report("Completed collecting ShakeNBreak results, workchain finished.")

    # Common methods:
    def setup_composition(self, structure_data: orm.StructureData):
        """Setup composition."""
        structure = structure_data.get_pymatgen_structure()
        composition = structure.composition.to_pretty_string()
        return composition

    def _determine_hpc(self):
        code = orm.load_code(self.inputs.code_string_vasp_gam.value)
        return code.computer.label

    def _setup_number_cores_per_machine(self, hpc_string: str):
        """Determine number of cores based on HPC"""
        if "archer" in hpc_string.lower():
            return 128
        elif "young" in hpc_string.lower():
            return 40
        else:
            raise ValueError("HPC string not recognised.")

    def _get_ncore(self, hpc_string: str):
        """Determine NCORE based on HPC chosen"""
        if "archer" in hpc_string.lower():
            return 8  # or 16
        elif "young" in hpc_string.lower():
            return 10

    def setup_options(
        self,
        hpc_string : str,
        num_machines: int,
        num_mpiprocs_per_machine: int=128,  # assume archer2
        num_cores_per_machine: int=128,  # assume archer2
        time_in_hours: int=10,
    ):
        """Setup options for HPC with aiida."""
        return setup_options(
            hpc_string=hpc_string,
            num_machines=num_machines,
            num_mpiprocs_per_machine=num_mpiprocs_per_machine,
            num_cores_per_machine=num_cores_per_machine,
            time_in_hours=time_in_hours
        )

    def setup_settings(self, calc_type: str="screening"):
        return setup_settings(calc_type)

    def parse_relaxed_structure(self, workchain) -> orm.StructureData:
        """Get output (relaxed) structure from workchain."""
        try:
            return workchain.outputs.relax.structure
        except:
            self.exit_codes.ERROR_OUTPUT_STRUCTURE_NOT_FOUND.format(
                pk=workchain.pk
            )

    def parse_final_energy(self, workchain) -> float:
        """Parse final energy from relaxation workchain."""
        try:
            return workchain.outputs.misc.get_dict()['total_energies']['energy_extrapolated']
        except:
            self.exit_codes.ERROR_PARSING_OUTPUT.format(
                pk=workchain.pk
            )

    def add_workchain_to_group(self, workchain, group_label):
        """Add workchain to group."""
        group_path = GroupPath()
        if group_label:
            group = group_path[group_label].get_or_create_group()
            group = group_path[group_label].get_group()
            group.add_nodes(workchain)
            # print(
            #     f"Submitted relax workchain with pk: {workchain.pk}"
            #     +f" and label {workchain.label}, "
            #     +f"stored in group with label {group_label}"
            # )
        # else:
        #     print(
        #         f"Submitted relax workchain with pk: {workchain.pk} and label {workchain.label}"
        #     )

    def validate_finished_workchain(self, workchain_name: str):
        """Validate that the workchain finished successfully."""
        if workchain_name not in self.ctx:
            raise RuntimeError(f"Workchain {workchain_name} not found in context.")

        workchain = self.ctx[workchain_name]
        cls = self._process_class.__name__
        # Check if workchain finished successfully:
        if not workchain.is_finished_ok:
            exit_status = self.ctx.workchain.exit_status
            self.report(
                f"{cls}<{self.workchain.pk}> (label={workchain.label}) failed with exit status {exit_status}."
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=cls,
                exit_status=exit_status
            )
        # All ok
        self.report(
            f'{cls}<{self.ctx.workchain.pk}> finished successfully.'
        )


class DefectsWorkChain(WorkChain, metaclass=ABCMeta):
    """Workflow to apply ShakeNBreak to all intrinsic defects
    for a certain host.

    Args:
        structure: StructureData
        defect_types: List
        screen_intersitials: Bool
        submit_relaxations: Bool
        symmetry_tolerance: Float
        supercell_min_length: Float
        supercell_max_number_atoms: Int
        supercell_min_number_atoms: Int
        charge_tolerance: Float
        code_string_vasp_gam: Str
        num_nodes: Int

    Returns:
        results (Dict)
        defect_entries_dict (Dict)
        snb_output_dicts (Dict):
            Dictionary with keys "distortions_dict" and "metadata_dict",
            storing the dictionaries output by ShakeNBreak.
    """

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.outline(
            cls.generate_defects,
            cls.parse_defects,
            if_(cls.should_run_screening)(
                cls.screen_interstitials,
                cls.parse_interstitials,
            ),
            cls.apply_shakenbreak,
            cls.finalize,
        )

        # Inputs:
        spec.expose_inputs(
            DefectGenerationWorkChain, namespace='defect_generation'
        )
        spec.input(
            "num_nodes",
            valid_type=orm.Int,
            required=False,
            default=orm.Int(1),
            help=("Number of nodes to use for relaxations. "
            "Recommended values: 1 using archer2, 2 if using Young "
            "(e.g. to get aorund 60-100 cores).")
        )
        # Code/HPC:
        spec.input(
            'code_string_vasp_gam',
            valid_type=orm.Str,
            required=False,
            default=orm.Str("vasp_gam_6.3.0"), # default is archer2
            help='Code string for vasp_gam',
        )
        spec.input(
            "num_nodes",
            valid_type=orm.Int,
            required=False,
            default=orm.Int(1),
            help=("Number of nodes to use for relaxations. "
            "Recommended values: 1 using archer2, 2 if using Young "
            "(e.g. to get aorund 60-100 cores).")
        )
        spec.input(
            "screen_interstitials",
            valid_type=orm.Bool,
            required=False,
            default=orm.Bool(True),
            help="Whether to screen interstitials."
        )
        spec.input(
            'submit_relaxations',
            valid_type=orm.Bool,
            required=False,
            default=orm.Bool(True), # default is archer2
            help='Whether to submit relaxations of defects.',
        )
        # Specify outputs:
        # Structures, Trajectories and forces for
        # all snb relaxations
        spec.expose_outputs(ShakeNBreakWorkChain)
        spec.output(
            "defect_entries_dict",
            valid_type=orm.Dict,
            required=False,
        )
        spec.output(
            "screened_defect_entries_dict",
            valid_type=orm.Dict,
            required=False,
        )
        # Exit codes:
        spec.exit_code(
            0,
            'NO_ERROR',
            message='the sun is shining',
        )
        spec.exit_code(
            400,
            'ERROR_SUB_PROCESS_FAILED',
            message="The `{cls}` workchain failed with exit status {exit_status}."
        )
        spec.exit_code(
            600,
            'ERROR_OUTPUT_STRUCTURE_NOT_FOUND',
            message="Couldnt parse output structure from workchain {pk}."
        )
        spec.exit_code(
            501,
            'ERROR_PARSING_OUTPUT',
            message="Couldnt parse output from workchain {pk}."
        )

    def should_run_screening(self):
        """Whether to screen interstitials"""
        defect_types = self.inputs.defect_generation.defect_types.get_list()
        if (
            "interstitials" in defect_types
            and self.inputs.screen_interstitials
        ):
            return True
        else:
            return False

    def generate_defects(self):
        """Generate defects."""

        defect_generation_workchain = self.submit(
            DefectGenerationWorkChain,
            **self.exposed_inputs(DefectGenerationWorkChain, "defect_generation")
        )
        self.to_context(
            defect_generation_workchain=defect_generation_workchain
        )

    def parse_defects(self):
        if "defect_generation_workchain" not in self.ctx:
            raise ValueError("DefectGenerationWorkChain not in context")

        if not self.ctx.defect_generation_workchain.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=DefectGenerationWorkChain.__name__,
                exit_status=self.ctx.defect_generation_workchain.exit_status
            )
        # All ok
        defects_dict_aiida = self.ctx.defect_generation_workchain.outputs.defect_entries_dict
        self.ctx.defects_Dict_aiida = defects_dict_aiida  # orm.Dict
        self.out("defect_entries_dict", defects_dict_aiida)

        # self.ctx.defects_dict = refactor_defects_dict(
        #     deepcopy(defects_dict_aiida.get_dict())
        # )  # as python dict, easier for postprocessing

    def screen_interstitials(self):
        """
        Screen interstitials. For each element, select the interstitials lower
        in energy (if difference > 1eV).
        Also used to identify if any initial interstitial structures
        relaxed to the same final one (in this case, only one is selected).

        The screening is performed by performing Gamma point relaxations
        (for a given element, select interstitials lower in energy).
        """
        interstitial_screening_workchain = self.submit(
            InterstitialScreeningWorkChain,
            defects_dict_aiida=self.ctx.defects_Dict_aiida,
            screen_intersitials=self.inputs.screen_interstitials,
        )
        self.to_context(
            interstitial_screening_workchain=interstitial_screening_workchain
        )

    def parse_interstitials(self):
        if "interstitial_screening_workchain" not in self.ctx:
            raise ValueError("InterstitialScreeningWorkChain not in context")
        if not self.ctx.interstitial_screening_workchain.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=InterstitialScreeningWorkChain.__name__,
                exit_status=self.ctx.interstitial_screening_workchain.exit_status
            )
        # All ok
        self.ctx.defects_Dict_aiida = self.ctx.interstitial_screening_workchain.outputs.screened_defect_entries_dict
        self.out("screened_defect_entries_dict", self.ctx.defects_Dict_aiida)

    def apply_shakenbreak(self):
        """Apply ShakeNBreak."""
        inputs = {
            "code_string_vasp_gam": self.inputs.code_string_vasp_gam,
            "num_nodes": self.inputs.num_nodes,
            "defects_dict_aiida": self.ctx.defects_Dict_aiida,
            "submit_relaxations": self.inputs.submit_relaxations,
        }
        shakenbreak_workchain = self.submit(
            ShakeNBreakWorkChain,
            **inputs,
        )
        self.to_context(shakenbreak_workchain=shakenbreak_workchain)
        # output_Dict = apply_shakenbreak(  # calcfunction
        #     defects_Dict=self.ctx.defects_Dict_aiida  # dict with all pmg objects as dicts
        # )
        # self.ctx.distorted_dict_aiida = output_Dict["distortions_dict"]  # this is a python dict
        # self.out("snb_output_dicts", output_Dict)  # Both distortions & metadata dict
        # Note that in distortions_dict, all pmg objects are dicts!
        # This distortions_dict is formatted like:
        # {defect_name: {
        #    "defect_site": Site_as_dict,
        #    "defect_type": Site_as_dict,
        #    "defect_multiplicity: ,
        #    "defect_supercell_site": ,
        #    "charges": {
        #        charge: {
        #             "structures": {
        #                "Unperturbed": Structure_as_dict,
        #                "distortions": {"Bond_Distortion_-60.0%": Structure,}
        #             }
        #        }
        #    }
        # }

    def finalize(self):
        """Append ShakeNBreak results."""
        if "shakenbreak_workchain" not in self.ctx:
            raise RuntimeError(f"Workchain shakenbreak_workchain not found in context.")
        if self.ctx.shakenbreak_workchain.is_finished_ok:
            self.out_many(
                self.exposed_outputs(self.ctx.shakenbreak_workchain, ShakeNBreakWorkChain)
            )
            return self.exit_codes.NO_ERROR
        else:
            self.out_many(
                self.exposed_outputs(self.ctx.shakenbreak_workchain, ShakeNBreakWorkChain)
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=ShakeNBreakWorkChain.__name__,
                exit_status=self.ctx.shakenbreak_workchain.exit_status,
            )

    # General methods:
    def setup_composition(self, structure_data: orm.StructureData):
        """Setup composition."""
        structure = structure_data.get_pymatgen_structure()
        composition = structure.composition.to_pretty_string()
        return composition

    def _determine_hpc(self):
        code = orm.load_code(self.inputs.code_string_vasp_gam.value)
        return code.computer.label

    def _setup_number_cores_per_machine(self, hpc_string: str):
        """Determine number of cores based on HPC"""
        if "archer" in hpc_string.lower():
            return 128
        elif "young" in hpc_string.lower():
            return 40
        else:
            raise ValueError("HPC string not recognised.")

    def _get_ncore(self, hpc_string: str):
        """Determine NCORE based on HPC chosen"""
        if "archer" in hpc_string.lower():
            return 8  # or 16
        elif "young" in hpc_string.lower():
            return 10

    def setup_options(
        self,
        hpc_string : str,
        num_machines: int,
        num_mpiprocs_per_machine: int=128,  # assume archer2
        num_cores_per_machine: int=128,  # assume archer2
        time_in_hours: int=12,
    ):
        """Setup options for HPC with aiida."""
        return setup_options(
            hpc_string=hpc_string,
            num_machines=num_machines,
            num_mpiprocs_per_machine=num_mpiprocs_per_machine,
            num_cores_per_machine=num_cores_per_machine,
            time_in_hours=time_in_hours
        )

    def setup_settings(self, calc_type: str="snb"):
        return setup_settings(calc_type)

    def parse_relaxed_structure(self, workchain) -> orm.StructureData:
        """Get output (relaxed) structure from workchain."""
        try:
            return workchain.outputs.relax.structure
        except:
            self.exit_codes.ERROR_OUTPUT_STRUCTURE_NOT_FOUND.format(
                pk=workchain.pk
            )

    def parse_final_energy(self, workchain) -> float:
        """Parse final energy from relaxation workchain."""
        try:
            return workchain.outputs.misc.get_dict()['total_energies']['energy_extrapolated']
        except:
            self.exit_codes.ERROR_PARSING_OUTPUT.format(
                pk=workchain.pk
            )

    def add_workchain_to_group(self, workchain, group_label):
        """Add workchain to group."""
        group_path = GroupPath()
        if group_label:
            group = group_path[group_label].get_or_create_group()
            group = group_path[group_label].get_group()
            group.add_nodes(workchain)
            # print(
            #     f"Submitted relax workchain with pk: {workchain.pk}"
            #     +f" and label {workchain.label}, "
            #     +f"stored in group with label {group_label}"
            # )
        # else:
        #     print(
        #         f"Submitted relax workchain with pk: {workchain.pk} and label {workchain.label}"
        #     )

    def validate_finished_workchain(self, workchain_name: str):
        """Validate that the workchain finished successfully."""
        if workchain_name not in self.ctx:
            raise RuntimeError(f"Workchain {workchain_name} not found in context.")

        workchain = self.ctx[workchain_name]
        cls = self._process_class.__name__
        # Check if workchain finished successfully:
        if not workchain.is_finished_ok:
            exit_status = self.ctx.workchain.exit_status
            self.report(
                f"{cls}<{self.workchain.pk}> (label={workchain.label}) failed with exit status {exit_status}."
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=cls,
                exit_status=exit_status
            )
        # All ok
        self.report(
            f'{cls}<{self.ctx.workchain.pk}> finished successfully.'
        )



# Below are the functions and calcfunctions used withing the above workchain:

@calcfunction
def parse_snb_workchain(workchain) -> orm.Dict:
    """Parse the output of the SNB workchain."""
    out_dict = {
        "pk": workchain.pk,
        "remote_folder": workchain.outputs.remote_folder,
        "structure": workchain.outputs.relax.structure,
        "final_energy": workchain.outputs.misc.get_dict()["total_energies"]["energy_extrapolated"],
        "energies": workchain.outputs.energies,
        "stress": workchain.outputs.stress,
        "forces": workchain.outputs.forces,
        "trajectory": workchain.outputs.trajectory,
    }
    return orm.Dict(out_dict)


@calcfunction
def apply_shakenbreak(defects_Dict: orm.Dict) -> orm.Dict:
    """Apply ShakeNBreak to defects dictionary"""
    # 1. Refactor from DefectEntry_as_dict -> DefectEntry
    defects_dict = refactor_defects_dict(defects_Dict.get_dict())
    # 2. Refactor to remove defect_type classification
    snb_defects = {
        k: v for d in defects_dict.values() for k, v in d.items()
    }
    dist = Distortions(defects=snb_defects)
    try:
        distorted_dict, metadata_dict = dist.apply_distortions()
        # distorted_dict is formatted like:
        # {defect_name: {
        #    "defect_site":,
        #    "defect_type":,
        #    "defect_multiplicity: ,
        #    "defect_supercell_site": ,
        #    "charges": {
        #        charge: {
        #             "structures": {
        #                "Unperturbed": Structure,
        #                "distortions": {"Bond_Distortion_-60.0%": Structure,}
        #             }
        #        }
        #    }
        # }
        # Refactor dict to be compatible with aiida:
        # (e.g. Structure -> dict ; Site -> dict)
        distorted_dict_aiida = refactor_distortion_dict(distorted_dict)
        return orm.Dict(dict={
            "distortions_dict": distorted_dict_aiida,
            "metadata_dict": metadata_dict
        })
    except Exception as e:
        raise RuntimeError(f"Exception {e} was raised while applying ShakeNBreak")


# Functions used to convert from dicts with Pymatgen objects to their dict equivalent (for aiida)

def refactor_defects_dict(defect_entries_dict: dict) -> dict:
    """Transform defects_dict so that
    DefectEntry_as_dict -> DefectEntry object"""
    for key, value_dict in defect_entries_dict.items():  # for defect type (e.g. vacancies)
        defect_entries_aiida = {}
        for defect_name, defect_entry_list in value_dict.items():
            defect_entries_aiida[defect_name] = []
            for defect_entry_as_dict in defect_entry_list:
                defect_entries_aiida[defect_name].append(
                    DefectEntry.from_dict(defect_entry_as_dict)
            )
        defect_entries_dict[key] = deepcopy(defect_entries_aiida)  # for each defect type, dict of DefectEntries
    return defect_entries_dict


def refactor_distortion_dict(
    distorted_dict: dict,
):
    """Refactor distorted_dict to be compatible with aiida
    (Structure -> Structure_as_dict, Site -> Site_as_dict)"""
    distorted_dict_aiida = {}
    for defect_name, dist_dict in distorted_dict.items():
        distorted_dict_aiida[defect_name] = {}

        # Add serilizable data
        for key, value in dist_dict.items():
            if "site" in key:
                distorted_dict_aiida[defect_name][key] = value.as_dict()
            elif key not in ["charges"]:
                distorted_dict_aiida[defect_name][key] = value

        # Refactor Structure -> Structure_as_dict
        distorted_dict_aiida[defect_name]["charges"] = {}
        for charge, charge_dict in dist_dict["charges"].items():
            distorted_dict_aiida[defect_name]["charges"][charge] = {
                "structures": {
                    "distortions": {},
                    "Unperturbed": charge_dict["structures"]["Unperturbed"].as_dict(),
                }
            }
            for dist_name, structure in (
                charge_dict["structures"]["distortions"].items()
            ):
                distorted_dict_aiida[defect_name]["charges"][charge]["structures"]["distortions"][
                    dist_name
                ] = structure.as_dict()

    return distorted_dict_aiida
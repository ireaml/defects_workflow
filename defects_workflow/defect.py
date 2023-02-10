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

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import  VaspInputSet

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
    generate_supercell_n_defects,
    sort_interstitials_for_screening
)

from defects_workflow.vasp_input import setup_incar_snb


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
path = GroupPath()

class DefectsWorkChain(WorkChain, metaclass=ABCMeta):
    """Workflow to apply ShakeNBreak to all intrinsic defects
    for a certain host."""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.outline(
            cls.generate_defects,
            if_(cls.should_run_screening)(cls.screen_interstitials, cls.analyse_screening_results),
            cls.apply_shakenbreak,
            cls.relax_defects,
            cls.parse_snb_relaxations,
            cls.results,
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
            "screen_interstitials",
            valid_type=orm.Bool,
            required=False,
            default=orm.Bool(True),
            help="Whether to screen interstitials."
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
        # Code/HPC:
        spec.input(
            'code_string_vasp_gam',
            valid_type=orm.Str,
            required=False,
            default=orm.Str("vasp_gam_6.3.0"), # default is archer2
            help='Code string for vasp_gam',
        )
        # Specify outputs:
        # Structures, Trajectories and forces for
        # all snb relaxations
        spec.output(
            'results',
            valid_type=orm.Dict,
            required=False,
        )
        spec.output(
            "defect_entries_dict",
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
            500,
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
        if (
            "interstitials" in self.inputs.defect_types
            and self.inputs.screen_interstitials
        ):
            return True
        else:
            return False

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


    def generate_defects(self):
        """Generate defects."""

        self.report("generate_defects")

        # Generate defects with pymatgen-analysis-defects
        relaxed_structure = self.inputs.structure
        defects_dict_aiida = generate_supercell_n_defects(
            bulk=relaxed_structure,
            defect_types=self.inputs.defect_types,
            min_length=self.inputs.supercell_min_length,
            symprec=self.inputs.symmetry_tolerance,
            min_atoms=orm.Int(30),
            max_atoms=self.inputs.supercell_max_number_atoms,
            force_diagonal=orm.Bool(False),
            interstitial_min_dist=orm.Float(1.0),
            dummy_species=orm.Str("X"),  # to keep track of frac coords in sc
            charge_tolerance=self.inputs.charge_tolerance,
        )  # type orm.Dict (not dict!)
        # In defects_dict_aiida, DefectEntries are stored as `dictionaries`,
        # should refactor to DefectEntrys here

        self.ctx.defects_dict_aiida = deepcopy(defects_dict_aiida)  # store for output
        self.out("defect_entries_dict", defects_dict_aiida)
        self.ctx.defects_dict = refactor_defects_dict(defects_dict_aiida).get_dict()  # as python dict, easier for postprocessing


    def screen_interstitials(self):
        """
        Screen interstitials. For each element, select the interstitials lower
        in energy (if difference > 1eV).
        Also used to identify if any initial interstitial structures
        relaxed to the same final one (in this case, only one is selected).

        The screening is performed by performing Gamma point relaxations
        (for a given element, select interstitials lower in energy).
        """
        self.report("screen_interstitials")

        # Get composition string
        self.ctx.composition = self.setup_composition(self.inputs.structure)

        # Get interstitials with sym ineq configurations (@calcfunction):
        self.ctx.int_dict = sort_interstitials_for_screening(
            orm.Dict(self.ctx.defects_dict)
        ).get_dict()  # as python dict

        # Get incar (same for all sym ineq interstitials)
        structure = list(list(self.ctx.int_dict.values())[0].values())[0].sc_entry.structure
        # Update ncore based on hpc of code:
        self.ctx.hpc_string = self._determine_hpc()
        self.ctx.ncore = self._get_ncore(hpc_string=self.ctx.hpc_string)
        # For screening, only relax neutral:
        defect_relax_set_dict = setup_incar_snb(
            supercell=orm.StructureData(pymatgen_structure=structure),
            charge=orm.Int(0),
            incar_settings=orm.Dict(
                {"NCORE": self.ctx.ncore}
            ),
        )
        incar = defect_relax_set_dict["incar"]

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
        for general_int_name, defect_entry_dict in self.ctx.int_dict.items():
            for defect_name, defect_entry in defect_entry_dict.items():
                structure = defect_entry.sc_entry.structure
                # Submit relaxation (gamma point):
                workchain, inputs = setup_relax_inputs(
                    code_string=self.inputs.code_string_vasp_gam.value,
                    # aiida config:
                    options=self.ctx.options,
                    settings=settings,
                    # VASP inputs:
                    structure_data=structure,
                    kpoints_data=self.ctx.gam_kpts, # 1,1,1
                    incar_dict=deepcopy(incar.as_dict()),
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
        self.report("analyse_interstitial_screening_results")

        # Loop over element speific interstitials: (Te_i, Cd_i)
        for defect_entry_dict in self.ctx.int_dict.values():
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
                    self.ctx.defects_dict["interstitials"].pop(defect_name, None)
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
                        self.ctx.defects_dict["interstitials"].pop(defect_name, None)


    def apply_shakenbreak(self):
        """Apply ShakeNBreak."""
        self.report("apply_shakenbreak")
        self.ctx.distorted_dict, metadata_dict = apply_shakenbreak(  # calcfunction
            defects_Dict=orm.Dict(self.ctx.defects_dict)
        )
        self.ctx.distorted_dict = self.ctx.distorted_dict.get_dict()  # as dict
        # This orm.Dict is formatted like:
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


    def relax_defects(self):
        """Submit geometry optimisations for the defects."""
        self.report("relax_defects")
        # Setup settings
        settings = self.setup_settings(calc_type="snb")
        distorted_dict = self.ctx.distorted_dict.get_dict()
        # Loop over defects, charge states & distortions:
        for defect_name, dist_dict in distorted_dict.items():
            for charge in dist_dict["charges"]:
                # Get incar of neutral state (for screening only need neutral)
                defect_relax_set_dict = setup_incar_snb(  # calcfunction
                    supercell=orm.StructureData(pymatgen=structure),
                    charge=orm.Int(charge),
                    incar_settings=orm.Dict(
                        {"NCORE": self.ctx.ncore}  # update ncore based on hpc of code
                    ),
                )
                incar = defect_relax_set_dict["incar"]
                # Submit unperturbed structure:

                # Submit each distortion:
                for dist_name, structure in zip(
                    # Names:
                    [
                        "Unperturbed",
                    ]
                    + list(
                        dist_dict["charges"][charge]["structures"][
                            "distortions"
                        ].keys()
                    ),
                    # Structures:
                    [dist_dict["charges"][charge]["structures"]["Unperturbed"]]
                    + list(
                        dist_dict["charges"][charge]["structures"][
                            "distortions"
                        ].values()
                    ),
                ):
                    # Submit relaxation (gamma point):
                    workchain, inputs = setup_relax_inputs(
                        code_string=self.inputs.code_string_vasp_gam.value,
                        # aiida config:
                        options=self.ctx.options,
                        settings=settings,
                        # VASP inputs:
                        structure_data=structure,
                        kpoints_data=self.ctx.gam_kpts, # 1,1,1
                        # Incar settings:
                        incar_dict=deepcopy(incar.as_dict()),
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
        distorted_dict = self.ctx.distorted_dict
        out_dict = {}
        for defect_name, dist_dict in distorted_dict.items():
            out_dict[defect_name] = {
                "defect_entries": self.ctx.defects_dict_aiida[defect_name],  # list with DefectEntry for all charge states
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
        self.out("results", self.ctx.out_dict)


# Below are the functions and calcfunctions used withing the above workchain
# See https://aiida.readthedocs.io/projects/aiida-core/en/v2.2.1/topics/
# workflows/concepts.html#topics-workflows-concepts-workchains

def refactor_defects_dict(defect_entries_dict: dict) -> dict:
    # Transform DefectEntry_as_dict -> DefectEntry object
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
    # Refactor to dict of DefectEntry's (e.g. remove defect_type classification)
    defects_dict = defects_Dict.get_dict()
    snb_defects = {
        k: v for d in defects_dict.values() for k, v in d.items()
    }
    dist = Distortions(defects=snb_defects)
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
    return orm.Dict(dict=distorted_dict), orm.Dict(dict=metadata_dict)


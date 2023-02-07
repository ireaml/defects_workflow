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

from aiida.engine import WorkChain, ToContext
from aiida import orm

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from defects_workflow.utils import query_materials_project, get_kpoints_from_density, get_options_dict, compare_structures
from defects_workflow.relaxation import submit_relaxation
from defects_workflow.defect_generation import generate_supercell_n_defects, sort_interstitials_for_screening
from defects_workflow.vasp_input import setup_incar_snb


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class DefectsWorkChain(WorkChain, metaclass=ABCMeta):
    """Workflow to apply ShakeNBreak to all intrinsic defects
    for a certain host."""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=False,
            help='The host structure.'
        )
        spec.input(
            'symmetry_tolerance',
            valid_type=orm.Float,
            required=False,
            help='Symmetry tolerance for space group analysis on the input structure.',
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
            default="vasp_gam_6.3.0", # default is archer2
            help='Code string for vasp_gam',
        )
        spec.outline(
            cls.generate_defects,
            cls.screen_interstitials,
            cls.analyse_screening_results,
            cls.apply_snb,
            cls.relax_defects,
        )
        spec.output('result', valid_type=orm.Int)
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

    def _determine_hpc(self):
        code = orm.load_code(self.inputs.code_string_vasp_gam.value)
        return code.computer.label

    def _setup_number_cores_per_machine(self, hpc_string: str):
        """Determine number of cores based on HPC"""
        if hpc_string == "archer2":
            return 128
        elif hpc_string == "young":
            return 80
        else:
            raise ValueError("HPC string not recognised.")

    def _get_ncore(self, hpc_string: str):
        """Determine NCORE based on HPC chosen"""
        if hpc_string == "archer2":
            return 8
        elif hpc_string == "young":
            return 10

    def setup_options(
        self,
        hpc_string : str,
        num_machines: int,
        num_mpiprocs_per_machine: int=128,  # assume archer2
        num_cores_per_machine: int=128,  # assume archer2
        time_in_hours: int=24,
    ):
        """Setup options for HPC with aiida."""
        options = get_options_dict(self.inputs.hpc_string)
        if hpc_string == "archer2":
            options.update({
                'resources':
                    {
                        'num_machines': num_machines,
                        'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
                        'num_cores_per_machine': num_cores_per_machine,
                    },
                'max_wallclock_seconds': int(time_in_hours*3600),
            })
        elif hpc_string == "young":
            options.update({
                'resources':
                    {
                        "tot_num_mpiprocs": int(num_machines * num_mpiprocs_per_machine),
                        "parallel_env": "mpi"
                    },
                'max_wallclock_seconds': int(time_in_hours*3600),
            })
        return options

    def setup_setings(self, calc_type: str="snb"):
        if calc_type == "snb":
            return {
                "parser_settings": {
                    "misc": [
                        "total_energies",
                        "maximum_force",
                        "maximum_stress",
                        "run_status",
                        "run_stats",
                        "notifications",
                    ],
                    "add_structure": True,  # retrieve structure and kpoints
                    'add_forces' : True,
                    "add_stress": True,
                    "add_energies": True,
                    'add_trajectory': True,
                    # Dont parse:
                    "add_dos": False,
                    'add_kpoints': False,
                    'add_bands': False,
                },
            }
        elif calc_type == "screening":
            return {
                "parser_settings": {
                    "misc": [
                        "total_energies",
                        "maximum_force",
                        "maximum_stress",
                        "run_status",
                        "run_stats",
                        "notifications",
                    ],
                    "add_structure": True,  # retrieve structure
                    # Dont parse:
                    'add_forces' : False,
                    "add_stress": False,
                    "add_energies": False,
                    'add_trajectory': False,
                    "add_dos": False,
                    'add_kpoints': False,
                    'add_bands': False,
                },
            }

    def validate_finished_workchain(self, workchain_name: str):
        """Validate that the workchain finished successfully."""
        workchain = self.ctx[workchain_name]
        cls = self._process_class.__name__
        # Check if workchain finished successfully:
        if not workchain.is_finished_ok:
            exit_status = self.ctx.workchain.exit_status
            self.report(f"{cls}<{self.workchain.pk}> failed with exit status {exit_status}.")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=cls,
                exit_status=exit_status
            )

        self.report(
            f'{cls}<{self.ctx.workchain.pk}> finished successfully.'
        )


    def generate_defects(self):
        """Generate defects."""

        self.report("generate_defects")

        # Generate defects with pymatgen-analysis-defects
        relaxed_structure = self.inputs.structure
        defects_dict = generate_supercell_n_defects(
            bulk=relaxed_structure,
            min_length=self.inputs.supercell_min_length,
            symprec=self.inputs.symmetry_tolerance,
            min_atoms=orm.Int(30),
            max_atoms=self.inputs.supercell_max_number_atoms,
            force_diagonal=orm.Bool(False),
            interstitial_min_dist=orm.Float(1.0),
            dummy_species=orm.Str("X"),  # to keep track of frac coords in sc
        )
        self.ctx.defects_dict = defects_dict


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
        # Get interstitials with sym ineq configurations:
        self.ctx.int_dict = sort_interstitials_for_screening(self.ctx.defects_dict)

        # Get incar (same for all sym ineq interstitials)
        structure = list(list(int_dict.values())[0].values())[0].sc_entry.structure
        # for screening, only neutral:
        defect_relax_set = setup_incar_snb(supercell=structure, charge=0)
        incar = defect_relax_set.incar
        # Update ncore based on hpc of code:
        self.ctx.hpc_string = self._determine_hpc()
        self.ctx.ncore = self._get_ncore(hpc_string=self.ctx.hpc_string)
        incar.update({"NCORE": self.ctx.ncore})

        # Setup KpointsData
        self.ctx.gam_kpts = orm.KpointsData().set_kpoints_mesh([1, 1, 1])

        # Submit geometry optimisations for all interstitials at the same time:
        for general_int_name, defect_entry_dict in self.ctx.int_dict.value.items():
            for defect_name, defect_entry in defect_entry_dict.items():
                structure = defect_entry.sc_entry.structure
                # Submit relaxation (gamma point):
                # Setup settings for screening
                settings = self.setup_setings(calc_type="screening")
                # Submit workchain:
                workchain = submit_relaxation(
                    code_string=self.inputs.code_string_vasp_gam,
                    # aiida config:
                    options=self.ctx.options,
                    settings=settings,
                    # VASP inputs:
                    structure_data=structure,
                    kpoints_data=self.ctx.gam_kpts, # 1,1,1
                    incar_dict=deepcopy(incar.as_dict()),
                    shape=False,
                    volume=False,
                    ionic_steps=300,
                    # Labels:
                    workchain_label=f"screen_{defect_name}",  # eg Te_i_s32
                    group_label=f"defects_db/{self.ctx.mp_id}/02_interstitials_screening",
                )
                key = f"screen.{defect_name}"
                self.to_context(**{key: workchain})


    def analyse_screening_results(self):
        """
        Analyse screening results.
        If any initial interstitial structures relaxed to the same final one, select only one.
        If energy difference between two interstitials is > 1eV, select the one with the lowest energy.
        """
        self.report("analyse_interstitial_screening_results")
        screened_defect_dict = deepcopy(self.ctx.int_dict.value)

        for int_name, defect_entry_list in self.ctx.int_dict.value.items():
            energies = {}  # defect_name: energy of interstitial
            for defect_name, defect_entry in enumerate(defect_entry_list):
                # Validate workchains
                key = f"screen.{defect_name}"
                self.validate_finished_workchain(workchain_name=key)
                # Parse final energy
                energy = self.ctx[key].outputs.misc.get_dict()['total_energies']['energy_extrapolated']
                energies[defect_name] = energy
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
                    # Get keys of structures with same energy:
                    other_key = [
                        k for k
                        in energies.keys()
                        if abs(energies[key] - energy) < 0.1 and k != defect_name
                    ]
                    # Get structures:
                    structure_1 = self.ctx[f"screen.{defect_name}"].outputs.relax.structure
                    structure_1 = self.ctx[f"screen.{other_key}"].outputs.relax.structure
                    if compare_structures(struct_1, struct_2):
                        self.ctx.defects_dict["interstitials"].pop(defect_name, None)


    def apply_shakenbreak(self):
        """Apply ShakeNBreak."""
        self.report("apply_shakenbreak")
        # Refactor to dict of defect
        snb_defects = {
            k: v for d in self.ctx.defects_dict.values() for k, v in d.items()
        }
        dist = Distortions(defects=snb_defects)
        distorted_dict, metadata_dict = dist.apply_distortions()
        self.ctx.distorted_dict = distorted_dict
        # This dict is formatted like:
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
        settings = self.setup_setings(calc_type="snb")
        for defect_name, dist_dict in self.ctx.distorted_dict.items():
            for charge in dist_dict["charges"]:
                # Get incar
                for dist_name, structure in dist_dict["charges"][charge]["structures"]["distortions"].items():
                    # Submit relaxation (gamma point):
                    # Submit workchain:
                    workchain = submit_relaxation(
                        code_string=self.inputs.code_string_vasp_gam,
                        # aiida config:
                        options=self.ctx.options,
                        settings=settings,
                        # VASP inputs:
                        structure_data=structure,
                        kpoints_data=self.ctx.gam_kpts, # 1,1,1
                        incar_dict=deepcopy(incar.as_dict()),
                        shape=False,
                        volume=False,
                        ionic_steps=300,
                        # Labels:
                        workchain_label=f"snb.{defect_name}.{charge}.{dist_name}",  # e.g. snb.Te_i_s32.0.Unperturbed
                        group_label=f"defects_db/{self.ctx.mp_id}/03_snb_relaxations",
                    )
                    key = f"snb.{defect_name}.{charge}.{dist_name}"
                    self.to_context(**{key: workchain})

    def validate_relaxations(self):
        """Validate the relaxations."""
        self.report("validate_relaxations")
        for defect_name, dist_dict in self.ctx.distorted_dict.items():
            for charge in dist_dict["charges"]:
                # Get incar
                for dist_name, structure in dist_dict["charges"][charge]["structures"]["distortions"].items():
                    key = f"snb.{defect_name}.{charge}.{dist_name}"
                    self.validate_finished_workchain(workchain_name=key)
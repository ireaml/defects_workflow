""""Workflow to:
1. Query mp-id from mp-database
2. Relax
"""
import os
from abc import ABCMeta
from copy import deepcopy
from monty.serialization import loadfn
import math

from aiida.engine import WorkChain, ToContext, append_
from aiida import orm
from aiida.common.exceptions import NotExistent
from aiida.tools.groups import GroupPath

from aiida_vasp.workchains.relax import RelaxWorkChain
from aiida_vasp.utils.workchains import compose_exit_code

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from defects_workflow.utils import (
    query_materials_project,
    get_kpoints_from_density,
    get_options_dict,
    compare_structures
)
from defects_workflow.relaxation import setup_relax_inputs
from defects_workflow.vasp_input import setup_incar_snb
from defects_workflow.vasp_toolkit.input import get_default_number_of_bands

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class BulkWorkChain(WorkChain, metaclass=ABCMeta):
    """Workflow to query mp-id from mp-database and relax."""
    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)
        spec.expose_outputs(RelaxWorkChain)

        spec.input(
            'mp_id',
            valid_type=orm.Str,
            required=False,
            default=None,
            help='The materials project id of the host material.'
        )
        spec.input(
            'api_key',
            valid_type=orm.Str,
            required=False,
            default=None,
            help='The materials project api key.'
        )
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=False,
            default=None,
            help='The host structure. If provided both structure and the mp_id, the structure will take priority.'
        )
        spec.input(
            "kpoint_density",
            valid_type=orm.Float,
            required=False,
            default=orm.Float(1000),
            help='K-points density for the relaxation.',
        )
        spec.input(
            'code_string_vasp_std',
            valid_type=orm.Str,
            required=False,
            default=orm.Str("vasp_std_6.3.0"),  # default is vasp_std in archer2
            help='Code string for vasp_std',
        )
        spec.outline(
            cls.query_mp,
            cls.relax_host_isif_3,
            cls.verify_next_workchain,
            cls.validate_finished_workchain,
            cls.relax_host_isif_1,
            cls.verify_next_workchain,
            cls.store_relaxed,
            cls.results,
            cls.finalize,
        )
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
            401,
            'ERROR_INPUT_STRUCTURE',
            message='Either mp_id or structure must be provided.'
        )
        spec.output(
            'structure',
            valid_type=orm.StructureData,
            required=False
        )

    def _determine_hpc(self) -> str:
        code = orm.load_code(self.inputs.code_string_vasp_std.value)
        return code.computer.label

    def _get_ncore(self, hpc_str: str) -> int:
        """Determine NCORE based on HPC chosen"""
        if hpc_str == "archer2":
            return 8
        elif hpc_str == "young":
            return 10

    def setup_options(
        self,
        hpc_string : str,
        num_machines: int,
        num_mpiprocs_per_machine: int=128,  # assume archer2
        num_cores_per_machine: int=128,  # assume archer2
        time_in_hours: int=24,
    ) -> dict:
        """Setup options for HPC with aiida."""
        options = get_options_dict(hpc_string)
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
            # SGE scheduler for Young, so need to use different keys:
            options.update({
                'resources':
                    {
                        "tot_num_mpiprocs": int(num_machines * num_mpiprocs_per_machine),
                        "parallel_env": "mpi"
                    },
                'max_wallclock_seconds': int(time_in_hours*3600),
            })
        return options

    def _determine_kpar_n_num_cores(
        self,
        structure_data: orm.StructureData,
        kpoints_data: orm.KpointsData,
        number_of_cores: orm.Int,
    ) -> tuple:
        """Determine KPAR based on the number of irreducible kpoints."""
        # Calculate VASP deafult number of bands
        num_bands = get_default_number_of_bands(structure_data.get_pymatgen_structure())
        # Calculate number of irr kpoints
        sa =  SpacegroupAnalyzer(structure_data.get_pymatgen_structure())
        num_irr_kpts = len(
            sa.get_ir_reciprocal_mesh(kpoints_data.get_kpoints_mesh()[0])
        )
        num_cores, kpar = max(
            {
                pos_num_cores: math.gcd(num_irr_kpts, pos_num_cores)
                for pos_num_cores in range(number_of_cores-5, number_of_cores+1)
            }.items(),
            key=lambda x: x[1]
        )
        return num_cores, kpar

    def _setup_number_cores_per_machine(self, hpc_str: str) -> int:
        """Determine number of cores based on HPC name."""
        if hpc_str == "archer2":
            return 128
        elif hpc_str == "young":
            return 80
        else:
            raise ValueError("HPC string not recognised.")

    def setup_setings(self) -> dict:
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
                "add_energies": True,
                # Dont parse:
                "add_stress": False,
                "add_forces": False,
                'add_trajectory': False,
                "add_dos": False,
                'add_kpoints': False,
                'add_bands': False,
            },
        }

    def parse_relaxed_structure(self, workchain):
        try:
            return workchain.outputs.relax.structure
        except:
            raise ValueError("Problem parsing relaxed structure.")

    def add_workchain_to_group(self, workchain, group_label):
        """Add workchain to group."""
        path = GroupPath()
        if group_label:
            group = path[group_label].get_or_create_group()
            group = path[group_label].get_group()
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

    def setup_composition(self, structure_data: orm.StructureData):
        """Setup composition."""
        structure = structure_data.get_pymatgen_structure()
        composition = structure.composition.to_pretty_string()
        return composition

    def query_mp(self):
        """Query Materials Project for the structure."""
        # Query Materials Project for the structure:
        if self.inputs.mp_id and not self.inputs.structure:
            self.ctx.structure = query_materials_project(
                api_key=self.inputs.api_key,
                material_id=self.inputs.mp_id,
            )
        elif self.inputs.structure:
            self.ctx.structure = self.inputs.structure

    def relax_host_isif_3(self):
        """Relax the host structure."""
        # Specify composition
        self.ctx.composition = self.setup_composition(
            structure_data=self.ctx.structure
        )
        # Setup VASP inputs:
        self.ctx.kpoints_data = get_kpoints_from_density(
            structure=self.ctx.structure,
            k_density=self.inputs.kpoint_density,
        )
        # Setup INCAR:
        incar_dict_host = loadfn(
            os.path.join(
                MODULE_DIR, "../yaml_files/vasp/incar/relax_host.yaml"
            )
        )
        # Setup HPC resources:
        # Determine HPC name from code_string
        self.ctx.hpc_string = self._determine_hpc()
        self.ctx.number_cores_per_machine = self._setup_number_cores_per_machine(
            hpc_str=self.ctx.hpc_string
        )
        # Determine KPAR based on number of irreducible kpoints & number of cores (#TODO)
        self.ctx.num_cores, self.ctx.kpar  = self._determine_kpar_n_num_cores(
            structure_data=self.ctx.structure,
            kpoints_data=self.ctx.kpoints_data,
            number_of_cores=self.ctx.number_cores_per_machine,
        )
        # Determine NCORE based on HPC chosen
        self.ctx.ncore = self._get_ncore(hpc_str=self.ctx.hpc_string)
        # Check NPAR (#TODO)
        # Update INCAR dict
        incar_dict_host.update(
            {
                "ISIF": 3,
                "ENCUT": 1.3 * incar_dict_host["ENCUT"],
                "KPAR": self.ctx.kpar,
                "NCORE": self.ctx.ncore
            }
        )
        self.ctx.options = self.setup_options(
            hpc_string=self.ctx.hpc_string,
            num_machines=2, # TODO
            num_cores_per_machine=self.ctx.num_cores,
            num_mpiprocs_per_machine=self.ctx.num_cores,
        )
        # Specify VASP output files that should be retrieved:
        self.ctx.settings = self.setup_setings()
        # Setup workchain:
        workchain, inputs = setup_relax_inputs(
            code_string=self.inputs.code_string_vasp_std.value,
            # aiida config:
            options=self.ctx.options,
            settings=self.ctx.settings,
            # VASP inputs:
            structure_data=self.ctx.structure,
            kpoints_data=self.ctx.kpoints_data,
            # Relaxation parameters:
            incar_dict=deepcopy(incar_dict_host),
            use_default_incar_settings=False,
            shape=True,
            volume=True,
            ionic_steps=300,
            # Labels:
            workchain_label="relax_host_isif_3",
        )
        # Submit the requested workchain with the supplied inputs
        workchain = self.submit(
            workchain,
            **inputs
        )
        # group_label=f"defects_db/{self.inputs.mp_id}/01_relax_host"
        # key = "host.relax.isif_3"
        # return ToContext(**{key: workchain})
        return self.to_context(workchains=append_(workchain))

    def validate_finished_workchain(self):
        """Validate that the workchain finished successfully."""
        #workchain = self.ctx[workchain_name]
        workchain = self.ctx.workchains[-1]
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

    def relax_host_isif_1(self):
        """Relax the host structure."""

        # Validate previous workchain:
        # self.validate_finished_workchain(workchain_name="host.relax.isif_3")

        # Parse relaxed structure from previous workchain:
        #workchain = self.ctx.host.relax.isif_3
        workchain = self.ctx.workchains[-1]
        relaxed_structure = self.parse_relaxed_structure(workchain=workchain)

        # Setup VASP input:
        incar_dict_host = loadfn(
            os.path.join(
                MODULE_DIR, "../yaml_files/vasp/incar/relax_host.yaml"
            )
        )
        incar_dict_host.update(
            {
                "KPAR": self.ctx.kpar,
                "NCORE": self.ctx.ncore,
                "ISIF": 2,
            }
        )
        # Submit workchain:
        workchain, inputs = setup_relax_inputs(
            code_string=self.inputs.code_string_vasp_std.value,
            # aiida config:
            options=self.ctx.options,
            settings=self.ctx.settings,
            # VASP inputs:
            structure_data=relaxed_structure,
            kpoints_data=self.ctx.kpoints_data,
            # Input parameters:
            incar_dict=deepcopy(incar_dict_host),
            use_default_incar_settings=False,
            shape=False,
            volume=False,
            ionic_steps=300,
            # Labels:
            workchain_label="relax_host_isif_1",
        )
        workchain = self.submit(workchain, **inputs)
        #key = "host.relax.isif_1"
        #group_label=f"defects_db/{self.inputs.mp_id}/01_relax_host",
        return self.to_context(workchains=append_(workchain))

    def verify_next_workchain(self):
        """Verify and inherit exit status from child workchains.
        Reproduced from aiida-vasp RelaxWorkChain.
        """

        try:
            workchain = self.ctx.workchains[-1]
        except IndexError:
            self.report(f'There is no {self._next_workchain.__name__} in the called workchain list.')
            return self.exit_codes.ERROR_NO_CALLED_WORKCHAIN  # pylint: disable=no-member

        # Inherit exit status from last workchain (supposed to be
        # successfull)
        next_workchain_exit_status = workchain.exit_status
        next_workchain_exit_message = workchain.exit_message
        if not next_workchain_exit_status:
            self.ctx.exit_code = self.exit_codes.NO_ERROR  # pylint: disable=no-member
        else:
            self.ctx.exit_code = compose_exit_code(
                next_workchain_exit_status,
                next_workchain_exit_message
            )
            self.report(
                f'The called {workchain.__class__.__name__}<{workchain.pk}> returned a non-zero exit status. '
                f'The exit status {self.ctx.exit_code} is inherited'
            )
            # Make sure at the very minimum we attach the misc node (if it exists) that contains notifications and other
            # quantities that can be salvaged
            try:
                self.out('misc', workchain.outputs['misc'])
            except NotExistent:
                pass

        return self.ctx.exit_code

    def store_relaxed(self):
        """Store the relaxed structure."""
        workchain = self.ctx.workchains[-1]

        relaxed_structure = self.parse_relaxed_structure(workchain=workchain)
        self.report(
            "attaching the node {}<{}> as '{}'".format(
                relaxed_structure.__class__.__name__,
                relaxed_structure.pk,
                'structure'
            )
        )
        self.out('structure', relaxed_structure)

    def results(self):
        """Attach the remaining output results."""
        self.out_many(
            self.exposed_outputs(self.ctx.host.relax.isif_1, RelaxWorkChain)
        )

    def finalize(self):
        """Finalize the workchain."""
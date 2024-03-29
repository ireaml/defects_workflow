""""Workflow to:
1. Query mp-id from mp-database
2. Relax
"""
import os
from abc import ABCMeta
from copy import deepcopy
from monty.serialization import loadfn
import math

from aiida.engine import WorkChain, ToContext, append_, calcfunction, if_
from aiida import orm
from aiida.common.exceptions import NotExistent
from aiida.tools.groups import GroupPath

from aiida_vasp.workchains.relax import RelaxWorkChain
from aiida_vasp.utils.workchains import compose_exit_code

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from defects_workflow.utils import (
    query_materials_project,
    get_kpoints_from_density,
    setup_options,
    setup_settings,
)
from defects_workflow.relaxation import setup_relax_inputs
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
            default=orm.Float(900),
            help='K-points density for the relaxation (as used in pymatgen `automatic_density` method).',
        )
        # HPC related inputs:
        spec.input(
            'code_string_vasp_std',
            valid_type=orm.Str,
            required=False,
            default=orm.Str("vasp_std_6.3.0"),  # default is vasp_std in archer2
            help='Code string for vasp_std',
        )
        spec.input(
            "priority",
            valid_type=orm.Bool,
            required=False,
            default=orm.Bool(False),
            help=("Whether to run calculations in priority/gold or free queue. ")
        )
        spec.input(
            "account",
            valid_type=orm.Str,
            required=False,
            default=None,
            help=("Account to use for the calculations. ")
        )
        spec.input(
            "time_in_hours",
            valid_type=orm.Float,
            required=False,
            default=orm.Float(24),
            help=("Time in hours to use for the calculations. ")
        )
        spec.input(
            "submit_isif_3",
            valid_type=orm.Bool,
            required=False,
            default=orm.Bool(True),
            help=("Whether to submit a volume relaxation (ISIF=3) or not. ")
        )
        spec.outline(
            cls.query_mp,
            cls.setup_vasp_inputs,
            if_(cls.should_submit_isif_3)(
                cls.relax_host_isif_3,
                cls.inspect_relax_isif_3,
            ).else_(
                cls.setup_isif_1,
            ),
            cls.relax_host_isif_1,
            cls.inspect_relax_isif_1,
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
        spec.exit_code(
            402,
            "ERROR_HPC_NOT_RECOGNISED",
            message="The computer label of the specified code was not recognised."
        )
        spec.exit_code(
            420,
            'ERROR_NO_CALLED_WORKCHAIN',
            message='no called workchain detected',
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
        spec.exit_code(
            502,
            "ERROR_RELAX_FAILURE",
            message="Initial relaxation has failed!"
        )
        spec.output(
            'structure',
            valid_type=orm.StructureData,
            required=False
        )

    def _determine_hpc(self) -> str:
        code = orm.load_code(self.inputs.code_string_vasp_std.value)
        return code.computer.label

    def _get_ncore(self, hpc_string: str) -> int:
        """Determine NCORE based on HPC chosen"""
        if "archer" in hpc_string.lower():
            return 8
        elif "young" in hpc_string.lower():
            return 10

    def setup_options(
        self,
        hpc_string : str,
        num_machines: int,
        num_mpiprocs_per_machine: int=128,  # assume archer2
        num_cores_per_machine: int=128,  # assume archer2
        time_in_hours: int=24,
        priority: bool=False,
        account: str=None,
    ) -> dict:
        """Setup options for HPC with aiida."""
        return setup_options(
            hpc_string=hpc_string,
            num_machines=num_machines,
            num_mpiprocs_per_machine=num_mpiprocs_per_machine,
            num_cores_per_machine=num_cores_per_machine,
            time_in_hours=time_in_hours,
            priority=priority,
            account=account,
        )

    def _determine_kpar_n_num_cores(
        self,
        structure_data: orm.StructureData,
        kpoints_data: orm.KpointsData,
        ncore: int,
        number_of_cores_per_machine: int,
    ) -> tuple:
        """Determine KPAR based on the number of irreducible kpoints."""
        # Calculate VASP deafult number of bands
        # num_bands = get_default_number_of_bands(structure_data.get_pymatgen_structure())
        # Calculate number of irr kpoints
        sa =  SpacegroupAnalyzer(structure_data.get_pymatgen_structure())
        num_irr_kpts = len(
            sa.get_ir_reciprocal_mesh(kpoints_data.get_kpoints_mesh()[0])
        )
        # Get a number of cores & KPAR so
        # that is KPAR is the Greatest Common Divisor of num_irr_kpts and num_cores_divide_by_NCORE
        cores_per_ncore = int(number_of_cores_per_machine // ncore)
        num_nodes, kpar = max(
            {
                num_nodes: math.gcd(num_irr_kpts, int(num_nodes * cores_per_ncore))
                for num_nodes in range(1, 3)
            }.items(),
            key=lambda x: x[1]
        )
        # Make sure KPAR not too large (mem issues)
        if kpar >= 16:
            if kpar % 2 == 0:
                kpar = kpar // 2
        # Check multiple of NPAR (not too many extra bands)
        # num_cores_per_kpar_ncore = num_cores / (kpar * ncore)
        return num_nodes, kpar

    def _setup_number_cores_per_machine(self, hpc_string: str) -> int:
        """Determine number of cores based on HPC name."""
        if "archer" in hpc_string.lower():
            return 128
        elif "young" in hpc_string.lower():
            return 40
        else:
            return self.exit_codes.ERROR_HPC_NOT_RECOGNISED

    def setup_setings(self) -> dict:
        return setup_settings(calc_type="relax_host")

    def parse_relaxed_structure(self, workchain):
        try:
            return workchain.outputs.relax.structure
        except:
            return self.exit_codes.ERROR_OUTPUT_STRUCTURE_NOT_FOUND

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
            self.report(
                f"Query has been successful. The composition for the mp-id {self.inputs.mp_id.value} "
                f"is {self.setup_composition(structure_data=self.ctx.structure)}"
            )
        elif self.inputs.structure:
            self.ctx.structure = self.inputs.structure

    def setup_kpoints(
        self,
        structure_data: orm.StructureData,
        kpoint_density: orm.Float
    ):
        """Setup kpoints."""
        kpoints_data = get_kpoints_from_density(
            structure=structure_data,
            k_density=kpoint_density,
        )
        self.report(
            f"Based on density {kpoint_density.value} the kpoints are {kpoints_data.get_kpoints_mesh()}"
        )
        return kpoints_data

    def setup_vasp_inputs(self):
        # Specify composition
        self.ctx.composition = self.setup_composition(
            structure_data=self.ctx.structure
        )
        # Setup VASP inputs: kpoints, incar
        self.ctx.kpoints_data = self.setup_kpoints(
            structure_data=self.ctx.structure,
            kpoint_density=self.inputs.kpoint_density,
        )
        # Setup HPC resources:
        # Determine HPC name from code_string
        self.ctx.hpc_string = self._determine_hpc()
        self.ctx.number_cores_per_machine = self._setup_number_cores_per_machine(
            hpc_string=self.ctx.hpc_string
        )
        # Determine NCORE based on HPC chosen
        self.ctx.ncore = self._get_ncore(hpc_string=self.ctx.hpc_string)
        # Determine KPAR based on number of irreducible kpoints & number of cores (#TODO)
        self.ctx.num_nodes, self.ctx.kpar  = self._determine_kpar_n_num_cores(
            structure_data=self.ctx.structure,
            kpoints_data=self.ctx.kpoints_data,
            number_of_cores_per_machine=self.ctx.number_cores_per_machine,
            ncore=self.ctx.ncore,
        )
        # Check NPAR (#TODO)

        # Specify VASP output files that should be retrieved:
        self.ctx.settings = self.setup_setings()

    def should_submit_isif_3(self):
        if self.inputs.submit_isif_3:
            return True
        else:
            return False

    def setup_isif_1(self):
        self.ctx.relaxed_structure = self.ctx.structure
        self.ctx.time_in_hours = self.inputs.time_in_hours

    def relax_host_isif_3(self):
        """Relax the host structure."""
        # Setup INCAR (specific for ISIF=3 relaxation):
        incar_dict_host = setup_host_incar(
            dict_user_params={
                "ISIF": 3,
                "KPAR": self.ctx.kpar,
                "NCORE": self.ctx.ncore
            }
        )
        incar_dict_host["ENCUT"] = 1.3 * incar_dict_host["ENCUT"]

        # Setup HPC options:
        self.ctx.options = self.setup_options(
            hpc_string=self.ctx.hpc_string,
            num_machines=self.ctx.num_nodes,
            num_cores_per_machine=self.ctx.number_cores_per_machine,
            num_mpiprocs_per_machine=self.ctx.number_cores_per_machine,
            time_in_hours=self.inputs.time_in_hours.value,
            priority=self.inputs.priority.value,
            account=self.inputs.account.value,
        )
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
        self.to_context(host_relax_isif_3=workchain)

    def inspect_relax_isif_3(self):
        """Validate that the workchain finished successfully."""
        if "host_relax_isif_3" not in self.ctx:
            return self.exit_codes.ERROR_NO_CALLED_WORKCHAIN  # pylint: disable=no-member
            # raise RuntimeError("Relaxation workchain not found in the context")

        workchain = self.ctx.host_relax_isif_3
        # Check if workchain finished successfully:
        if not workchain.is_finished_ok:
            exit_status = workchain.exit_status
            self.report(
                f"Relaxation ISIF=3 (pk={workchain.pk}) finished with error, "
                f"exit status: {exit_status}."
            )
            return self.exit_codes.ERROR_RELAX_FAILURE  # pylint: disable=no-member

        # All OK
        self.ctx.relaxed_structure = self.parse_relaxed_structure(workchain=workchain)
        self.report(
            f"Relaxation ISIF=3 finished OK (pk={workchain.pk}). "
        )

    def relax_host_isif_1(self):
        """Relax the host structure."""

        # Setup VASP input:
        incar_dict_host = setup_host_incar(
            dict_user_params={
                "KPAR": self.ctx.kpar,
                "NCORE": self.ctx.ncore,
                "ISIF": 2,
            }
        )
        # Setup HPC options:
        if "time_in_hours" not in self.ctx:
            self.ctx.time_in_hours = 10

        self.ctx.options = self.setup_options(
            hpc_string=self.ctx.hpc_string,
            num_machines=self.ctx.num_nodes,
            num_cores_per_machine=self.ctx.number_cores_per_machine,
            num_mpiprocs_per_machine=self.ctx.number_cores_per_machine,
            time_in_hours=self.ctx.time_in_hours,  # small, cause already relaxed with ISIF=3
            priority=self.inputs.priority.value,
            account=self.inputs.account.value,
        )
        # Submit workchain:
        workchain, inputs = setup_relax_inputs(
            code_string=self.inputs.code_string_vasp_std.value,
            # aiida config:
            options=self.ctx.options,
            settings=self.ctx.settings,
            # VASP inputs:
            structure_data=self.ctx.relaxed_structure,
            kpoints_data=self.ctx.kpoints_data,
            # Input parameters:
            incar_dict=deepcopy(incar_dict_host),
            use_default_incar_settings=False,
            shape=False,
            volume=False,
            # Labels:
            workchain_label="relax_host_isif_1",
        )
        workchain = self.submit(workchain, **inputs)
        #group_label=f"defects_db/{self.inputs.mp_id}/01_relax_host",
        self.to_context(host_relax_isif_3=workchain)

    def inspect_relax_isif_1(self):
        """Validate that the relaxation finished successfully."""
        if "host_relax_isif_1" not in self.ctx:
            return self.exit_codes.ERROR_NO_CALLED_WORKCHAIN  # pylint: disable=no-member
            # raise RuntimeError("Relaxation workchain not found in the context")

        workchain = self.ctx.host_relax_isif_1
        # Check if workchain finished successfully:
        if not workchain.is_finished_ok:
            exit_status = workchain.exit_status
            self.report(
                f"Relaxation ISIF=1 (pk={workchain.pk}) finished with error, "
                f"exit status: {exit_status}."
            )
            return self.exit_codes.ERROR_RELAX_FAILURE  # pylint: disable=no-member

        # All OK
        self.ctx.relaxed_structure = self.parse_relaxed_structure(workchain=workchain)
        self.report(
            f"Relaxation ISIF=1 finished OK (pk={workchain.pk}). "
        )

    def store_relaxed(self):
        """Store the relaxed structure."""
        self.report(
            "attaching the node {}<{}> as '{}'".format(
                self.ctx.relaxed_structure.__class__.__name__,
                self.ctx.relaxed_structure.pk,
                'structure'
            )
        )
        self.out('structure', self.ctx.relaxed_structure)

    def results(self):
        """Attach the remaining output results."""
        self.out_many(
            self.exposed_outputs(self.ctx.host_relax_isif_1, RelaxWorkChain)
        )

    def finalize(self):
        """Finalize the workchain."""



# Functions
def setup_host_incar(
    dict_user_params: dict = None,
):
    """Load default INCAR and update with user
    defined parameters.
    """
    incar_dict_host = loadfn(
        os.path.join(
            MODULE_DIR, "yaml_files/vasp/incar/relax_host.yaml"
        )
    )
    incar_dict_host.update(dict_user_params)
    return incar_dict_host
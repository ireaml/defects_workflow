""""Workflow to:
1. Query mp-id from mp-database
2. Relax
3. Generate defects
4. Screen interstitials or defects in sym. ineq. sites
5. Apply SnB
"""
import os
from abc import ABCMeta
from copy import deepcopy
from monty.serialization import loadfn

from aiida.engine import WorkChain, ToContext
from aiida import orm

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from utils import query_materials_project, get_kpoints_from_density, get_options_dict
from relaxation import submit_relaxation

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


class DefectsWorkChain(WorkChain, metaclass=ABCMeta):
    """Workflow to apply ShakeNBreak to all intrinsic defects
    for a certain host."""

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        spec.input(
            'mp_id',
            valid_type=(orm.Str),
            required=False,
            help='The materials project id of the host material.'
        )
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=False,
            help='The host structure. If provided both structure and the mp_id, the structure will take priority.'
        )
        spec.input(
            'api_key',
            valid_type=orm.Str,
            required=False,
            default="MsKnfQSzWAraK6zyhZ7OlNTlVl2GMuWr",
            help='The key to access the MP api.'
        )
        spec.input(
            "kpoint_density",
            valid_type=orm.Float,
            required=False,
            default=orm.Float(1000),
            help="The kpoint density to use for the relaxation of the host structure."
        )
        spec.input(
            'symmetry_tolerance',
            valid_type=orm.Float,
            required=False,
            help='Symmetry tolerance for space group analysis on the input structure.',
        )
        spec.input(
            'code_string_vasp_std',
            valid_type=orm.Str,
            required=False,
            default="vasp_std_6.3.0",  # Check this!
            help='Code string for vasp_std',
        )
        spec.input(
            'hpc_string',
            valid_type=orm.Str,
            required=False,
            default="archer2",  # Check this!
            help='Code string specifying the HPC to use.',
        )
        spec.outline(
            cls.query_mp,
            cls.relax_host,
            cls.generate_defects,
            cls.screen_interstitials,
            cls.apply_snb,
            cls.relax_defects,
        )
        spec.output('result', valid_type=orm.Int)
        spec.exit_code(400, 'ERROR_INPUT_STRUCTURE', message='Either mp_id or structure must be provided.')


    def query_mp(self):
        """Query the materials project database for the structure of the host material."""
        if 'structure' in self.inputs:
            self.ctx.structure = self.inputs.structure
        elif 'mp_id' in self.inputs:
            structure = query_materials_project(
                material_id=self.inputs.mp_id, api_key=self.inputs.api_key
            )
            return ToContext(structure=structure)
        else:
            return self.exit_codes.ERROR_INPUT_STRUCTURE

    def _determine_kpar(
        self,
        structure_data: orm.StructureData,
        kpoints_data: orm.KpointsData,
        #number_of_cores: orm.Int,
    ):
        """Determine KPAR based on the number of irreducible kpoints."""
        # Calculate number of irr kpoints
        sa =  SpacegroupAnalyzer(structure_data.get_pymatgen_structure())
        num_irr_kpts = len(
            sa.get_ir_reciprocal_mesh(kpoints_data.get_kpoints_mesh()[0])
        )
        # If num_irr_kpts = 5 -> kpar=5
        # If num_irr_kpts = 4 -> kpar=4
        # If num_irr_kpts = 3 -> kpar=3
        # If num_irr_kpts = 2 -> kpar=2
        if num_irr_kpts % 5 == 0:
            return num_irr_kpts
        elif num_irr_kpts % 4 == 0:
            return num_irr_kpts
        elif  num_irr_kpts % 3 == 0:
            return num_irr_kpts
        elif num_irr_kpts % 2 == 0:
            return num_irr_kpts

    def _get_ncore(self):
        """Determine NCORE based on HPC chosen"""
        if self.inputs.hpc_string== "archer2":
            return 8
        elif self.inputs.hpc_string == "young":
            return 10

    def relax_host_isif_3(self):
        """Relax the host structure."""
        # Setup VASP input:
        self.ctx.kpoints_data = get_kpoints_from_density(
            structure=self.ctx.structure,
            k_density=self.inputs.kpoint_density,
        )
        incar_dict_host = loadfn(
            os.path.join(
                MODULE_DIR, "yaml_files/vasp/incar/relax_host.yaml"
            )
        )
        incar_dict_host["ISIF"] = 3
        incar_dict_host["ENCUT"] = 1.3 * incar_dict_host["ENCUT"]
        # Setup HPC resources:
        # Determine KPAR based on number of irreducible kpoints (#TODO)
        self.ctx.kpar = self._determine_kpar(
            structure_data=self.ctx.structure,
            kpoints_data=self.ctx.kpoints_data,
        )
        # Determine NCORE based on HPC chosen
        self.ctx.ncore = self._get_ncore()
        incar_dict_host.update(
            {"KPAR": self.ctx.kpar, "NCORE": self.ctx.ncore}
        )
        # Check NPAR (#TODO)
        options = get_options_dict(self.inputs.hpc_string)
        options.update({
            'resources':
                {
                    'num_machines': 2,
                    'num_mpiprocs_per_machine': 128,
                    'num_cores_per_machine': 128,
                },
            'max_wallclock_seconds': int(24*3600),
        })
        # Specify VASP output files that should be retrieved:
        settings = {
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
                # 'add_kpoints': True,
                # 'add_forces' : True,
                "add_energies": True,
                # 'add_bands': True,
                # 'add_trajectory': True,
            },
        }
        # Setup workchain:
        workchain = submit_relaxation(
            code_string=self.inputs.code_string_vasp_std,
            # aiida config:
            options=options,
            settings=settings,
            # VASP inputs:
            structure_data=self.ctx.structure,
            kpoints_data=self.ctx.kpoints_data,
            incar_dict=deepcopy(incar_dict_host),
            shape=True,
            volume=True,
            ionic_steps=300,
            # Labels:
            workchain_label="relax_host_isif_3",
            group_label=f"defects_db/{self.ctx.mp_id}",
        )
        return ToContext(wc_relax_host_isif_3=workchain)

    def validate_finished_workchain(self):
        """Validate that the workchain finished successfully."""
        workchain = self.ctx.wc_relax_host_isif_3
        # Check if workchain finished successfully:


    def relax_host_isif_1(self):
        """Relax the host structure."""
        # Parse relaxed structure from previous workchain:
        workchain = self.ctx.wc_relax_host_isif_3
        # Setup VASP input:
        incar_dict_host = loadfn(
            os.path.join(
                MODULE_DIR, "yaml_files/vasp/incar/relax_host.yaml"
            )
        )
        incar_dict_host.update(
            {"KPAR": self.ctx.kpar, "NCORE": self.ctx.ncore, "ISIF": 1}
        )
        workchain = submit_relaxation(
            code_string=self.inputs.code_string_vasp_std,
            # aiida config:
            options=options,
            settings=settings,
            # VASP inputs:
            structure_data=self.ctx.structure,
            kpoints_data=self.ctx.kpoints_data,
            incar_dict=deepcopy(incar_dict_host),
            shape=True,
            volume=True,
            ionic_steps=300,
            # Labels:
            workchain_label="relax_host_isif_3",
            group_label=f"defects_db/{self.ctx.mp_id}",
        )
        return ToContext(wc_relax_host_isif_1=workchain)

    def generate_defects(self):
        """Generate defects."""


    def screen_interstitials(self):
        """Screen interstitials."""


    def apply_shakenbreak(self):
        """Apply ShakeNBreak."""


    def relax_defects(self):
        """Submit geometry optimisations for the defects."""







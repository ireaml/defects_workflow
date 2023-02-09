# defects_workflow
Workflow to run defect calculations with aiida.

Currently, it automatises the following steps:
1. Relaxation of host structure (from mp-id or user defined structure)
2. Defect generation
   * Defect charge states are determined based on the most common oxidation states for the element (following
     the strategy implemented in [defectivator](https://github.com/alexsquires/defectivator)
     by Dr Alex Squires)
1. Screening of symmetry inequivalent interstitials.
   This is done by relaxing the neutral state of all the symmetry inequivalent
   configurations for a given interstitial. The following cases are filtered out:
    * Configurations that lead to the same final structures (only one is used for later calculations)
    * Configurations very high in energy compared to the most stable one (e.g. if > 1 eV)
2. Structure searching using shakenbreak and submission of calculations
3. Post-processing of results
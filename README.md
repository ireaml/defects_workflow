# defects_workflow
Workflow to run defect calculations with aiida.

Currently, it automatises the following steps:
1. Relaxation of host structure (from mp-id or user defined structure)
2. Defect generation
3. Screening of symmetry inequivalent interstitials. This is done by relaxing (the neutral state)
    all the symmetry inequivalent configurations for given interstitial. The following cases are filtered out:
    * Configurations that lead to same final structures (only one is used for later calculations)
    * Configurations very high in energy compared to the most stable one (e.g. if > 1 eV)
4. Structure searching using shakenbreak and submission of calculations
5. Post-processing of results
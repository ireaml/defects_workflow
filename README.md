# `defects_workflow`
Workflow to run defect calculations with aiida.

Currently, it automates the following steps:
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


# Installation

1. Crate conda environment (python 3.10)

2. Install `aiida-core` using the [system-wide installation](https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/install_system.html#intro-get-started-system-wide-install) and using `pip` rather than `conda`.

3. Install other dependencies, including `aiida-archer2-scheduler` (to user `archer2`),
    `parsevasp`, `aiida-vasp`, `aiida-user-addons` and `defectivator`:
```
git clone git@github.com:SMTG-UCL/aiida-archer2-scheduler.git
cd aiida-archer2-scheduler
pip install -e ./
reentry scan -r aiida
```
```
git clone https://github.com/aiida-vasp/parsevasp.git
cd parsevasp
git checkout develop
cd ../
pip install -e ./parsevasp
```
```
git clone https://github.com/aiida-vasp/aiida-vasp.git
cd aiida-vasp
git checkout develop
cd ../
pip install -e ./aiida-vasp
```
```
git clone https://github.com/SMTG-UCL/aiida-user-addons.git
cd aiida-user-addons
git checkout dev
cd ../
pip install -e ./aiida-user-addons
```
```
git clone https://github.com/alexsquires/defectivator.git
cd defectivator
git checkout dev
cd ../
pip install ./defectivator
```

Run `pip install reentry`
And `reentry scan -r aiida`

4. Configure aiida-vasp (potcars)

5. Install `shakenbreak`
```
git clone https://github.com/SMTG-UCL/shakenbreak.git
cd shakenbreak
git checkout dimer
cd ../
pip install ./shakenbreak
```

1. Install `defcets_workflow`
```
git clone <github_path>
pip install .
```

1. Add ab-initio codes to aiida profile

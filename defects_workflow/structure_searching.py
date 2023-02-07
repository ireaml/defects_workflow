"""Functions to apply distortions to defects using ShakeNBreak."""

from __future__ import annotations
import numpy as np
import warnings
from copy import deepcopy

from aiida.orm import Dict, Float, Int, Bool, StructureData, ArrayData

from shakenbreak.input import Distortions

def apply_shakenbreak(
    defect_entries_dict: dict,
    distortion_increment: Float=Float(0.1),
    stdev: Float=Float(0.25),
):
    # Refactor defect_entries_dict to be a list of DefectEntries rather
    # than a dict
    defect_entries_list = sum(defect_entries_dict.values(), [])
    # Apply distortions:
    dist = Distortions(
        defects=defect_entries_list,
        distortion_increment=distortion_increment,
        stdev=stdev,
    )
    distorted_dict, metadata_dict = dist.apply_distortions()
    # Now we need to get INCAR dict for each defect
    # Need to homogenize potcar keywords of setup_incar_snb
    # and the potcar keywords of the workchain
    return Dict(
        {"distortions_dict": Dict(distorted_dict), "metadata_dict": metadata_dict}
    )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:14:18 2026

@author: temuuleu


"""

from slurm_lib import submit_slurm_job

import os



LOCAL_LLM  = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/"

cmd_str = "python /sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/fine_tune_llm.py"

fine_tune_log = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/fine_tune_log"
slurm_scripts = "/sc-projects/sc-proj-cc15-csb-neuroimaging/SC_Stroke_MRI/temuuleu/Courses/LOCAL_LLM/slurm_scripts"


os.makedirs(fine_tune_log, exist_ok=True)
os.makedirs(slurm_scripts, exist_ok=True)

job_id = submit_slurm_job(
    cmd_str=cmd_str,
    job_name="fine_tune_llm",
    log_dir=fine_tune_log,
    slurm_dir=slurm_scripts,
    time_limit="48:00:00",  # anpassen nach Bedarf
    memory="64G",
    gpus="gpu:nvidia_a100_80gb_pcie:1",
    conda_env_path = "/sc-software/conda_envs/envs/gpulab-2025-2"
)


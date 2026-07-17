import sys
from submit_vertex_job import submit_job

job_name = "snaps_dyn_clear_n1000_vol0_7"
extra_args = [
    "--text", "COMP",
    "--steps", "16000",
    "--log-every", "100",
    "--no-noise",
    "--update-every", "1000",
    "--support-vol", "0.7",
]

submit_job("nca/train_dynamic_organic.py", extra_args=extra_args, job_name=job_name)

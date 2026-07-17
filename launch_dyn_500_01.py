import sys
from submit_vertex_job import submit_job

job_name = "snaps_dyn_clear_n500_vol0_1"
extra_args = [
    "--text", "COMP",
    "--steps", "16000",
    "--log-every", "100",
    "--no-noise",
    "--update-every", "500",
    "--support-vol", "0.1",
]

submit_job("nca/train_dynamic_organic.py", extra_args=extra_args, job_name=job_name)

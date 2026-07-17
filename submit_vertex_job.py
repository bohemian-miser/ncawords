import argparse
import os
import re
import subprocess
import sys

try:
    from google.cloud import aiplatform
    from google.cloud import storage
except ImportError:
    print("Please install google-cloud-aiplatform first: pip install google-cloud-aiplatform")
    sys.exit(1)

PROJECT_ID = "recipe-lanes-staging"
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
BUCKET_NAME = "recipe-lanes-nca-jobs"
STAGING_BUCKET = f"gs://{BUCKET_NAME}"
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"


def build_and_upload_package():
    print("Building source package for 'nca'...")
    subprocess.run([sys.executable, "setup.py", "sdist", "--formats=gztar"], check=True)
    tar_path = "dist/nca-0.1.tar.gz"

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("packages/nca-0.1.tar.gz")
    blob.upload_from_filename(tar_path)
    print(f"Uploaded package to {STAGING_BUCKET}/packages/nca-0.1.tar.gz")
    return f"{STAGING_BUCKET}/packages/nca-0.1.tar.gz"


def check_required_args(script_path, job_args):
    """Fail fast if the script declares required argparse flags we aren't passing.

    Vertex spot jobs can queue for 15+ minutes before running; an argparse
    error at that point wastes the whole wait. Catch it at submit time.
    """
    with open(script_path) as f:
        source = f.read()
    missing = []
    for match in re.finditer(r'add_argument\(\s*["\'](--[\w-]+)["\'][^)]*required\s*=\s*True', source):
        flag = match.group(1)
        if not any(a == flag or a.startswith(flag + "=") for a in job_args):
            missing.append(flag)
    if missing:
        raise SystemExit(
            f"Refusing to submit: {script_path} requires {missing} but they were not provided. "
            f"Pass them after the script path, e.g. --text COMP"
        )


# GPU tiers. Tiny-grid NCA training is kernel-launch-bound, so measure
# before assuming a bigger card helps (see the t4-vs-l4 benchmark runs).
MACHINES = {
    "t4":   {"machine_type": "n1-standard-4",  "accelerator_type": "NVIDIA_TESLA_T4"},
    "l4":   {"machine_type": "g2-standard-4",  "accelerator_type": "NVIDIA_L4"},
    "a100": {"machine_type": "a2-highgpu-1g",  "accelerator_type": "NVIDIA_TESLA_A100"},
}


def submit_job(script_path, extra_args=None, job_name=None, on_demand=False,
               location=LOCATION, package_uri=None, machine="t4"):
    aiplatform.init(project=PROJECT_ID, location=location, staging_bucket=STAGING_BUCKET)

    if package_uri is None:
        package_uri = build_and_upload_package()

    if not job_name:
        job_name = f"nca-train-{os.path.basename(script_path).split('.')[0]}"

    clean_path = script_path.replace("\\", "/").rstrip("/")
    if clean_path.endswith(".py"):
        clean_path = clean_path[:-3]
    module_name = clean_path.replace("/", ".")

    job_args = list(extra_args or [])
    if not any(a.startswith("--snap-dir") for a in job_args):
        job_args.append(f"--snap-dir=/gcs/{BUCKET_NAME}/{job_name}")

    check_required_args(script_path, job_args)

    print(f"Submitting module '{module_name}' to Vertex AI CustomJob ({job_name})...")
    print(f"  args: {job_args}")
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=job_name,
        python_package_gcs_uri=package_uri,
        python_module_name=module_name,
        container_uri=CONTAINER_URI,
    )

    strategy = aiplatform.compat.types.custom_job.Scheduling.Strategy.SPOT
    if on_demand:
        print("Submitting as ON-DEMAND job (instant, but full price).")
        strategy = aiplatform.compat.types.custom_job.Scheduling.Strategy.STANDARD
    else:
        print("Submitting as SPOT job (queued & discounted).")

    spec = MACHINES[machine]
    job.run(
        args=job_args,
        machine_type=spec["machine_type"],
        accelerator_type=spec["accelerator_type"],
        accelerator_count=1,
        scheduling_strategy=strategy,
        sync=False,
    )

    print(f"\nQueued {job_name}.")
    print(f"Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"Outputs: {STAGING_BUCKET}/{job_name}")
    return job


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Submit a training script to Vertex AI for GPU training. "
                    "Unrecognised flags are passed through to the training script, "
                    "e.g.: python submit_vertex_job.py nca/train_web_hidden.py --text COMP --steps 500")
    p.add_argument("script", help="Path to your training script (e.g., nca/train_web_hidden.py)")
    p.add_argument("--on-demand", action="store_true",
                   help="Run on an On-Demand instance instead of Spot (costs more but starts immediately)")
    p.add_argument("--job-name", default=None, help="Override the Vertex display name / output dir name")
    p.add_argument("--location", default=LOCATION, help=f"Vertex region (default {LOCATION})")
    p.add_argument("--machine", default="t4", choices=sorted(MACHINES),
                   help="GPU tier (default t4)")
    args, passthrough = p.parse_known_args()

    if not os.path.exists(args.script):
        print(f"Error: Could not find script '{args.script}'")
        sys.exit(1)

    submit_job(args.script, extra_args=passthrough, job_name=args.job_name,
               on_demand=args.on_demand, location=args.location,
               machine=args.machine)

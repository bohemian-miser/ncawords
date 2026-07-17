import argparse
import sys
import os

try:
    from google.cloud import aiplatform
except ImportError:
    print("Please install google-cloud-aiplatform first: pip install google-cloud-aiplatform")
    sys.exit(1)

PROJECT_ID = "recipe-lanes-staging"
LOCATION = os.environ.get("VERTEX_LOCATION", "us-east1")
STAGING_BUCKET = "gs://recipe-lanes-nca-jobs" 

import subprocess
from google.cloud import storage

def build_and_upload_package():
    print("Building source package for 'nca'...")
    subprocess.run([sys.executable, "setup.py", "sdist", "--formats=gztar"], check=True)
    tar_path = "dist/nca-0.1.tar.gz"
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket("recipe-lanes-nca-jobs")
    blob = bucket.blob("packages/nca-0.1.tar.gz")
    blob.upload_from_filename(tar_path)
    print("Uploaded package to gs://recipe-lanes-nca-jobs/packages/nca-0.1.tar.gz")
    return f"{STAGING_BUCKET}/packages/nca-0.1.tar.gz"

def submit_job(script_path, extra_args=None, job_name=None, spot=True, on_demand=False, location="us-central1"):
    aiplatform.init(project=PROJECT_ID, location=location, staging_bucket=STAGING_BUCKET)

    package_uri = build_and_upload_package()

    CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"

    if not job_name:
        job_name = f"nca-train-{os.path.basename(script_path).split('.')[0]}"
    
    clean_path = script_path.replace("\\", "/").rstrip("/")
    if clean_path.endswith(".py"):
        clean_path = clean_path[:-3]
    module_name = clean_path.replace("/", ".")

    job_args = [f"--snap-dir=/gcs/recipe-lanes-nca-jobs/{job_name}"]
    if extra_args:
        job_args.extend(extra_args)

    print(f"Submitting module '{module_name}' to Vertex AI CustomJob ({job_name})...")
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=job_name,
        python_package_gcs_uri=package_uri,
        python_module_name=module_name,
        container_uri=CONTAINER_URI,
    )

    strategy = aiplatform.compat.types.custom_job.Scheduling.Strategy.SPOT
    if on_demand:
        print("Submitting as ON-DEMAND job (Instant, but full price).")
        strategy = aiplatform.compat.types.custom_job.Scheduling.Strategy.STANDARD
    else:
        print("Submitting as SPOT job (Queued & discounted).")

    job.run(
        args=job_args,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        scheduling_strategy=strategy,
        sync=False,
    )
    
    print(f"\nSuccessfully queued {job_name}!")
    print(f"You can monitor it via the Google Cloud Console (Vertex AI -> Training) or at:")
    print(f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"Your outputs and models will appear magically in the bucket: {STAGING_BUCKET}/{job_name}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Submit a local Python script to Vertex AI for GPU training")
    p.add_argument("script", help="Path to your training script (e.g., nca/train_adaptive_cloud_stepped.py)")
    p.add_argument("--on-demand", action="store_true", help="Run on an On-Demand instance instead of Spot (costs more but starts immediately)")
    args = p.parse_args()

    if not os.path.exists(args.script):
        print(f"Error: Could not find script '{args.script}'")
        sys.exit(1)

    submit_job(args.script, on_demand=args.on_demand)

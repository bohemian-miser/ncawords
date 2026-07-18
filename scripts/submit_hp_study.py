"""Launch a Vertex AI Hyperparameter Tuning (Vizier) study on the
ladder_seed recipe: Bayesian search over the joint space the star sweep
probes one-at-a-time. Trials report 'loss' via cloudml-hypertune (shipped
in the package's install_requires)."""
import sys
sys.path.insert(0, ".")
from google.cloud import aiplatform_v1
from submit_vertex_job import (PROJECT_ID, BUCKET_NAME, STAGING_BUCKET,
                               CONTAINER_URI, MACHINES, build_and_upload_package)

REGION = "us-central1"
MAX_TRIALS = 32
PARALLEL = 6

pkg = build_and_upload_package()

spec = MACHINES["l4"]
trial_job_spec = {
    "worker_pool_specs": [{
        "machine_spec": {
            "machine_type": spec["machine_type"],
            "accelerator_type": spec["accelerator_type"],
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100},
        "python_package_spec": {
            "executor_image_uri": CONTAINER_URI,
            "package_uris": [pkg],
            "python_module": "nca.train_ladder_seed",
            "args": ["--text", "COMP", "--steps", "6000", "--log-every", "200",
                     "--normal-p", "1.0", "--damage-occasional"],
            # NOTE: no --snap-dir — trials are throwaway; metric is the output
        },
    }],
    "scheduling": {"strategy": aiplatform_v1.types.Scheduling.Strategy.SPOT},
}

P = aiplatform_v1.types.StudySpec.ParameterSpec
study_spec = {
    "metrics": [{"metric_id": "loss", "goal": "MINIMIZE"}],
    "parameters": [
        {"parameter_id": "lr", "scale_type": "UNIT_LOG_SCALE",
         "double_value_spec": {"min_value": 3e-4, "max_value": 8e-3}},
        {"parameter_id": "hidden_n",
         "integer_value_spec": {"min_value": 32, "max_value": 160}},
        {"parameter_id": "channel_n",
         "discrete_value_spec": {"values": [12, 16, 20, 24]}},
        {"parameter_id": "batch",
         "discrete_value_spec": {"values": [8, 16, 32]}},
        {"parameter_id": "fire_rate",
         "double_value_spec": {"min_value": 0.25, "max_value": 1.0}},
        {"parameter_id": "pool_size",
         "discrete_value_spec": {"values": [64, 256]}},
    ],
    # Vizier's default Bayesian algorithm; median automated early stopping
    "measurement_selection_type": "LAST_MEASUREMENT",
    "median_automated_stopping_spec": {"use_elapsed_duration": False},
}

client = aiplatform_v1.JobServiceClient(
    client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"})
job = client.create_hyperparameter_tuning_job(
    parent=f"projects/{PROJECT_ID}/locations/{REGION}",
    hyperparameter_tuning_job={
        "display_name": "hpstudy-ladder-seed",
        "study_spec": study_spec,
        "max_trial_count": MAX_TRIALS,
        "parallel_trial_count": PARALLEL,
        "max_failed_trial_count": 10,
        "trial_job_spec": trial_job_spec,
    })
print(f"Created study: {job.name}")
print(f"{MAX_TRIALS} trials, {PARALLEL} parallel, spot L4 in {REGION}")

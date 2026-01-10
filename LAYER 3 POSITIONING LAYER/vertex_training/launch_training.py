"""
Launch Vertex AI training job.
"""

from google.cloud import aiplatform
from datetime import datetime

# Initialize Vertex AI
aiplatform.init(
    project='himari-opus-2',
    location='us-central1',
    staging_bucket='gs://himari-rl-models'
)

# Create custom training job
job = aiplatform.CustomContainerTrainingJob(
    display_name=f'himari-rl-training-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    container_uri='gcr.io/himari-opus-2/rl-trainer:latest',
    model_serving_container_image_uri='gcr.io/himari-opus-2/rl-trainer:latest',
)

print("Submitting training job to Vertex AI...")
print("This will take 8-10 hours with GPU")
print("")

# Run training job
model = job.run(
    replica_count=1,
    machine_type='n1-standard-4',       # 4 vCPUs, 15 GB RAM
    accelerator_type='NVIDIA_TESLA_T4',  # T4 GPU
    accelerator_count=1,
    args=['--num-episodes', '1000', '--device', 'cuda'],
    environment_variables={
        'AIP_STORAGE_URI': 'gs://himari-rl-models'
    },
    sync=True  # Wait for completion
)

print("")
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Model resource name: {model.resource_name}")
print(f"Model saved to: gs://himari-rl-models/final-models/ppo_latest.pt")
print("")

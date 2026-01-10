#!/usr/bin/env python3
"""
HIMARI Layer 3 - Vertex AI Deployment Script
Builds Docker container and submits training job to Vertex AI
"""

import os
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_ID = "himari-opus-position-layer"
REGION = "us-central1"
BUCKET_NAME = "himari-rl-models"
IMAGE_NAME = "himari-rl-trainer"
IMAGE_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")


def run_command(cmd: str, cwd: Path = None) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)"""
    print(f"\n>>> {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=os.sys.stderr)
    
    return result.returncode, result.stdout, result.stderr


def build_and_push_image(project_root: Path) -> str:
    """Build Docker image and push to Google Container Registry"""
    image_uri = f"gcr.io/{PROJECT_ID}/{IMAGE_NAME}:{IMAGE_TAG}"
    
    print("="*60)
    print("Step 1: Building Docker Image")
    print("="*60)
    
    # Build the image
    dockerfile_path = project_root / "vertex_ai" / "docker" / "Dockerfile"
    build_cmd = f"docker build -f {dockerfile_path} -t {image_uri} {project_root}"
    
    returncode, _, _ = run_command(build_cmd)
    if returncode != 0:
        raise RuntimeError("Docker build failed")
    
    print(f"\n✓ Image built: {image_uri}")
    
    print("\n" + "="*60)
    print("Step 2: Pushing to Container Registry")
    print("="*60)
    
    # Configure Docker for GCR
    configure_cmd = f"gcloud auth configure-docker"
    run_command(configure_cmd)
    
    # Push the image
    push_cmd = f"docker push {image_uri}"
    returncode, _, _ = run_command(push_cmd)
    if returncode != 0:
        raise RuntimeError("Docker push failed")
    
    print(f"\n✓ Image pushed: {image_uri}")
    
    return image_uri


def submit_training_job(image_uri: str, num_episodes: int = 1000) -> str:
    """Submit custom training job to Vertex AI"""
    print("\n" + "="*60)
    print("Step 3: Submitting Vertex AI Training Job")
    print("="*60)
    
    job_name = f"himari-rl-training-{IMAGE_TAG}"
    
    # Create training job configuration
    training_config = {
        "displayName": job_name,
        "containerSpec": {
            "imageUri": image_uri,
            "args": [
                "--bucket-name", BUCKET_NAME,
                "--model-dir", "models/himari-rl",
                "--num-episodes", str(num_episodes),
                "--save-interval", "100"
            ]
        },
        "machineSpec": {
            "machineType": "n1-standard-4",
            "acceleratorType": "NVIDIA_TESLA_T4",
            "acceleratorCount": 1
        },
        "scheduling": {
            "timeout": "7200s",  # 2 hours
            "restartJobOnWorkerRestart": True
        }
    }
    
    # Save config to temp file
    config_path = Path("/tmp/training_config.json")
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Submit job using gcloud
    submit_cmd = f"""gcloud ai custom-jobs create \
        --region={REGION} \
        --display-name={job_name} \
        --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri={image_uri},replica-count=1 \
        --args=--bucket-name={BUCKET_NAME},--model-dir=models/himari-rl,--num-episodes={num_episodes},--save-interval=100
    """
    
    returncode, stdout, _ = run_command(submit_cmd)
    if returncode != 0:
        raise RuntimeError("Failed to submit training job")
    
    print(f"\n✓ Training job submitted: {job_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    
    return job_name


def main():
    """Main deployment workflow"""
    parser = argparse.ArgumentParser(description='Deploy HIMARI RL training to Vertex AI')
    
    parser.add_argument(
        '--project-root',
        type=str,
        default='.',
        help='Path to project root directory'
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1000,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip Docker build/push (use existing image)'
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    
    print("\n" + "="*60)
    print("HIMARI Layer 3 - Vertex AI Deployment")
    print("="*60)
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Episodes: {args.num_episodes}")
    print("="*60)
    
    try:
        # Build and push image
        if not args.skip_build:
            image_uri = build_and_push_image(project_root)
        else:
            image_uri = f"gcr.io/{PROJECT_ID}/{IMAGE_NAME}:latest"
            print(f"Using existing image: {image_uri}")
        
        # Submit training job
        job_name = submit_training_job(image_uri, args.num_episodes)
        
        print("\n" + "="*60)
        print("✓ Deployment Complete!")
        print("="*60)
        print(f"Job Name: {job_name}")
        print(f"Image: {image_uri}")
        print(f"\nMonitor your training job at:")
        print(f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
        print(f"\nModels will be saved to:")
        print(f"gs://{BUCKET_NAME}/models/himari-rl/")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
HIMARI Layer 3 - Complete GCP Deployment Script
Orchestrates the full deployment process from training to API deployment.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_ID = "himari-opus-position-layer"
REGION = "us-central1"
BUCKET_NAME = "himari-rl-models"

# Container images
TRAINER_IMAGE = f"gcr.io/{PROJECT_ID}/himari-rl-trainer"
API_IMAGE = f"gcr.io/{PROJECT_ID}/himari-rl-api"


def run(cmd: str, cwd: str = None) -> bool:
    """Run command and return success status."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def deploy_training():
    """Deploy training job to Vertex AI."""
    print("\n" + "="*60)
    print("PHASE 1: Vertex AI Training Deployment")
    print("="*60)
    
    project_root = Path(__file__).parent
    
    # Build trainer image
    print("\nStep 1: Building trainer Docker image...")
    dockerfile = project_root / "vertex_ai" / "docker" / "Dockerfile"
    
    if not run(f'docker build -f "{dockerfile}" -t {TRAINER_IMAGE}:latest "{project_root}"'):
        print("ERROR: Docker build failed")
        return False
    
    # Push to GCR
    print("\nStep 2: Pushing to Container Registry...")
    if not run(f'docker push {TRAINER_IMAGE}:latest'):
        print("ERROR: Docker push failed")
        return False
    
    # Submit training job
    print("\nStep 3: Submitting Vertex AI training job...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    cmd = f'''gcloud ai custom-jobs create \
        --region={REGION} \
        --display-name=himari-rl-training-{timestamp} \
        --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri={TRAINER_IMAGE}:latest,replica-count=1 \
        --args=--bucket-name={BUCKET_NAME},--model-dir=models/himari-rl,--num-episodes=1000,--save-interval=100'''
    
    if not run(cmd):
        print("ERROR: Training job submission failed")
        return False
    
    print("\n✓ Training job submitted successfully!")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    return True


def deploy_api():
    """Deploy inference API to Cloud Run."""
    print("\n" + "="*60)
    print("PHASE 2: Cloud Run API Deployment")
    print("="*60)
    
    project_root = Path(__file__).parent
    cloud_run_dir = project_root / "cloud_run"
    
    # Build API image using Cloud Build
    print("\nStep 1: Building API image with Cloud Build...")
    
    # Create cloudbuild.yaml
    cloudbuild_yaml = f"""
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'cloud_run/Dockerfile'
      - '-t'
      - '{API_IMAGE}:latest'
      - '.'
images:
  - '{API_IMAGE}:latest'
"""
    
    cloudbuild_path = project_root / "cloudbuild.yaml"
    with open(cloudbuild_path, 'w') as f:
        f.write(cloudbuild_yaml)
    
    if not run(f'gcloud builds submit --config cloudbuild.yaml .', cwd=str(project_root)):
        print("ERROR: Cloud Build failed")
        return False
    
    # Deploy to Cloud Run
    print("\nStep 2: Deploying to Cloud Run...")
    
    cmd = f'''gcloud run deploy himari-rl-api \
        --image {API_IMAGE}:latest \
        --platform managed \
        --region {REGION} \
        --allow-unauthenticated \
        --memory 1Gi \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 10 \
        --timeout 60s \
        --set-env-vars GCS_BUCKET={BUCKET_NAME},GCS_MODEL_PATH=models/himari-rl/checkpoint.pt'''
    
    if not run(cmd):
        print("ERROR: Cloud Run deployment failed")
        return False
    
    print("\n✓ API deployed successfully!")
    return True


def setup_monitoring():
    """Setup Cloud Monitoring dashboards and alerts."""
    print("\n" + "="*60)
    print("PHASE 4: Cloud Monitoring Setup")
    print("="*60)
    
    project_root = Path(__file__).parent
    
    # Run monitoring setup script
    if not run(f'python monitoring/gcp_monitoring_setup.py', cwd=str(project_root)):
        print("WARNING: Monitoring setup had issues, continuing...")
        return True  # Non-critical, continue anyway
    
    print("\n✓ Monitoring setup complete!")
    return True


def test_api():
    """Test the deployed API."""
    print("\n" + "="*60)
    print("Testing Deployed API")
    print("="*60)
    
    # Get API URL
    result = subprocess.run(
        f'gcloud run services describe himari-rl-api --region={REGION} --format="value(status.url)"',
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print("ERROR: Could not get API URL")
        return False
    
    api_url = result.stdout.strip()
    print(f"\nAPI URL: {api_url}")
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    if not run(f'curl -s {api_url}/health'):
        print("ERROR: Health check failed")
        return False
    
    # Test prediction
    print("\n\nTesting prediction endpoint...")
    test_state = "[0.85, 0, 0, 1, 1, 0, 0, 0, 0.3, 1, 0.02, 0.015, 0.023, 0.018, 0.65, 0.2]"
    cmd = f'''curl -s -X POST {api_url}/predict \
        -H "Content-Type: application/json" \
        -d '{{"state": {test_state}}}'
    '''
    
    if not run(cmd):
        print("ERROR: Prediction test failed")
        return False
    
    print("\n\n✓ API tests passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='HIMARI GCP Deployment')
    parser.add_argument('--phase', type=str, choices=['training', 'api', 'monitoring', 'test', 'all'],
                       default='all', help='Deployment phase to run')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HIMARI Layer 3 - GCP Deployment")
    print("="*60)
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Bucket: {BUCKET_NAME}")
    print("="*60)
    
    success = True
    
    if args.phase in ['training', 'all']:
        if not deploy_training():
            success = False
            if args.phase != 'all':
                return 1
    
    if args.phase in ['api', 'all']:
        if not deploy_api():
            success = False
            if args.phase != 'all':
                return 1
    
    if args.phase in ['monitoring', 'all']:
        if not setup_monitoring():
            success = False
    
    if args.phase in ['test', 'all']:
        if not test_api():
            success = False
    
    print("\n" + "="*60)
    if success:
        print("✓ Deployment Complete!")
    else:
        print("⚠ Deployment completed with some issues")
    print("="*60)
    
    print(f"\nUseful Links:")
    print(f"- Vertex AI Training: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"- Cloud Run: https://console.cloud.google.com/run?project={PROJECT_ID}")
    print(f"- Cloud Storage: https://console.cloud.google.com/storage/browser/{BUCKET_NAME}?project={PROJECT_ID}")
    print(f"- Monitoring: https://console.cloud.google.com/monitoring/dashboards?project={PROJECT_ID}")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

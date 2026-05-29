#!/bin/bash
#SBATCH --job-name=sent_an               # Job name
#SBATCH --output=logs/sent_an_%j.out     # Standard output log (the %j is replaced by the job ID)
#SBATCH --error=logs/sent_an_%j.err      # Standard error log
#SBATCH --partition=dgxh100              # Partition to submit to
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=8                # Request 8 CPU cores
#SBATCH --mem=64G                        # Request 64GB of memory
#SBATCH --time=12:00:00                  # Maximum time the job will run (HH:MM:SS)

echo "Running on node: $(hostname)"
echo "Starting aggregation job at: $(date)"

IMAGE_FILE="pytorcher2.sif"

if [ ! -f "$IMAGE_FILE" ]; then
    echo "Error: Apptainer image not found at $IMAGE_FILE!"
    exit 1
fi

echo "Initiating Sentiment Analysis aggregation pipeline..."

apptainer exec --nv "$IMAGE_FILE" python sent_aggregation.py

echo "Job finished at: $(date)"

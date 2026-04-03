#!/bin/bash
# Run srun
srun \
  --job-name=puffer_soccer_auto \
  --nodes=1 \
  --cpus-per-task=32 \
  --mem=64G \
  --gres=gpu:1 \
  --time=01:00:00 \
  --account=torch_pr_45_tandon_advanced \
  --mail-type=END,FAIL \
  --mail-user=fyy2003@nyu.edu \
  --chdir=/scratch/fyy2003/repos/Puffer-Soccer \
  bash
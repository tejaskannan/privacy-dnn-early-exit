#!/bin/sh
#
#SBATCH --mail-user=tkannan@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tkannan/Documents/privacy-dnn-early-exit/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tkannan/Documents/privacy-dnn-early-exit/slurm/out/%j.%N.stderr
#SBATCH --partition=fast
#SBATCH --job-name=stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=800

python dataset_stats.py --dataset-name uci_har --window-sizes 10

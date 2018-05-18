#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=rf
#SBATCH --mail-type=END
#SBATCH --mail-user=ay963@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge
module load python/intel/2.7.6
RUNDIR=$SCRATCH/refugees/
mkdir -p $RUNDIR
  
DATADIR=$HOME/refugees
cd $RUNDIR
python run_rf(1,0,0,0)
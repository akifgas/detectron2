#!/bin/bash
#SBATCH -J tik4             
#SBATCH -A agasi             
#SBATCH -o tik4.out    
#SBATCH -p debug          
#SBATCH -N 1              
#SBATCH -n 1              
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=10       
#SBATCH --time=00:05:00      

module purge 
module load centos7.9/lib/cuda/11.3 

eval "$(/truba/home/agasi/miniconda3/bin/conda shell.bash hook)"
conda activate tik4 



python detectron2.py


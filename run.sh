#!/bin/bash
#SBATCH --job-name=MultiSupCon_celeba_60_freeze_40
#SBATCH --time=40:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --chdir=/home/wolodja5/
#SBATCH --output=/home/wolodja5/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12341
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

### the command to run
srun python traincl.py --data=/home/wolodja5/celeba \
--data-name=CELEBA --model-name=tresnet_l --image-size=64 \
--workers=4 --freeze=True --batch-size=56 --epochs=40 \
--batch-size_con=512 --epochs_con=40 --linear=True \
--learning_rate_con=0.5 --temp=0.1 --cosine \
--num-classes=40 --run=4

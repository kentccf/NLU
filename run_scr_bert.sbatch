#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=144:00:00
#SBATCH --job-name="nlu_project"
#SBATCH --mail-user=zy879@nyu.edu


module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0


NETID=zy879
source activate /scratch/${NETID}/nlu/env



cd /scratch/${NETID}/nlu/code/transformers
export PYTHONPATH=/scratch/${NETID}/nlu/code/transformers/src:$PYTHONPATH
python ./examples/run_glue.py \
    --model_type bert\
    --model_name_or_path bert-large-uncased \
    --task_name SCR \
    --do_train \
    --do_eval \
    --data_dir /scratch/zy879/nlu/data/SCR \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=4   \
    --per_gpu_train_batch_size=4   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /scratch/zy879/nlu/project/SCR_bert_run2/


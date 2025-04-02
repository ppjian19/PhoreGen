#!/bin/bash


export CUDA_VISIBLE_DEVICES=3
date +"%Y-%m-%d %H:%M:%S"
echo "Current Hostname: $(hostname)"
echo "Current GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Current PID for this sbatch job: $$ (not python job PID)"

task_name=sample_test

## Setting model log file
log_file=./logs/${task_name}.log
echo Logging to ${log_file}

outdir=./results/${task_name}/
check_point=./ckpt/crossdocked_pdbbind_trained.pt
phore_file_list=./data/phores_for_sampling/file_index.json
echo Checkpoint: ${check_point}

pos_guidance_opt='[{"type":"atom_prox","min_d":1.0,"max_d":3.0},{"type":"center_prox"}]'

python -u sample_all.py \
    --config ./configs/train_dock-cpx-phore.yml \
    --num_samples 100 \
    --batch_size 30 \
    --outdir ${outdir} \
    --check_point ${check_point} \
    --phore_file_list ${phore_file_list} \
    --add_edge predicted \
    --pos_guidance_opt ${pos_guidance_opt} \
    --sample_nodes_mode normal \
    --normal_scale 6.0 \
    --save_traj_prob 0.0 > ${log_file} 2>&1


echo Sampling finished!
date +"%Y-%m-%d %H:%M:%S"

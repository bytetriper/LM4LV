#!/bin/bash
numgpu=4
exp=$1
now=$(date +"%Y%m%d_%H%M%S")

ckpt_dir=../ckpt
stage_1_ckpt_path=$2
datasetpath=$3
save_path=$4
echo "ckpt_dir: ${ckpt_dir}"
echo "stage_1_ckpt_path: ${stage_1_ckpt_path}"
echo "datasetpath: ${datasetpath}"
rm ${ckpt_dir}/${exp} -r
mkdir -p ${ckpt_dir}/${exp}/log_rest/
mkdir -p ${save_path}
cp ./config/eval_MAE.yaml ${ckpt_dir}/${exp}/log_rest/train_${now}.yaml
deepspeed --include localhost:0 finetuning/ft_MAE.py \
    --stage 2 \
    --mode eval \
    --cfg ./config/eval_MAE.yaml \
    --data_path ${datasetpath} \
    --delta_ckpt_path ${stage_1_ckpt_path} \
    --device cuda \
    --save_path  ${save_path} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
    
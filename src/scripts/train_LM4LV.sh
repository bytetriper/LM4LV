#!/bin/bash
numgpu=4
exp=$1
now=$(date +"%Y%m%d_%H%M%S")

ckpt_dir=../ckpt
vision_config_path=$2
num_vision_token=$3
stage_1_ckpt_path=$4
datasetpath=$5
llm_ckpt_path=$6
echo "ckpt_dir: ${ckpt_dir}"
echo "vision_config_path: ${vision_config_path}"
echo "num_vision_token: ${num_vision_token}"
echo "stage_1_ckpt_path: ${stage_1_ckpt_path}"
echo "datasetpath: ${datasetpath}"
rm ${ckpt_dir}/${exp} -r
mkdir -p ${ckpt_dir}/${exp}/log_rest/
cp ./config/train_bf16.yaml ${ckpt_dir}/${exp}/log_rest/train_${now}.yaml
cp ${vision_config_path} ${ckpt_dir}/${exp}/log_rest/vision_config_${now}.yaml
deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 11451 train.py \
    --stage 2 \
    --cfg ./config/train_bf16.yaml \
    --vision_config_path ${vision_config_path} \
    --data_path ${datasetpath} \
    --vision_root_path ooo \
    --delta_ckpt_path ${stage_1_ckpt_path} \
    --conv_template default \
    --max_tgt_len 1024 \
    --device cuda \
    --model lamm_peft \
    --llm_ckpt_path ${llm_ckpt_path} \
    --num_vision_token ${num_vision_token} \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    --train_deadapter \
    --train_adapter \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
    
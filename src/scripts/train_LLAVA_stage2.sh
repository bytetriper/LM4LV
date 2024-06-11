#!/bin/bash
numgpu=4

exp=$1
visfeat_type=local
now=$(date +"%Y%m%d_%H%M%S")

ckpt_dir=../ckpt
vision_config_path=$2
num_vision_token=$3
delta_ckpt_path=$4
data_path=$5
llm_ckpt_path=$6
GPT_PREFIX='A picture of'
rm ${ckpt_dir}/${exp} -r
mkdir -p ${ckpt_dir}/${exp}/log_rest/
cp ./config/train_bf16.yaml ${ckpt_dir}/${exp}/log_rest/train_${now}.yaml
cp ${vision_config_path} ${ckpt_dir}/${exp}/log_rest/vision_config_${now}.yaml
deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 28457 train.py \
    --stage 2 \
    --cfg ./config/train_bf16.yaml \
    --vision_config_path ${vision_config_path} \
    --data_path ${data_path} \
    --vision_root_path ooo \
    --conv_template default \
    --max_tgt_len 1024 \
    --device cuda \
    --model lamm_peft \
    --llm_ckpt_path ${llm_ckpt_path} \
    --delta_ckpt_path ${delta_ckpt_path} \
    --num_vision_token ${num_vision_token} \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    --gpt_prefix "${GPT_PREFIX}" \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
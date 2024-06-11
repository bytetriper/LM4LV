#!/bin/bash
numgpu=4
exp=$1
now=$(date +"%Y%m%d_%H%M%S")
vision_config_path=$2
ckpt_dir=../ckpt
stage_1_ckpt_path=$3
datasetpath=$4

echo "ckpt_dir: ${ckpt_dir}"
echo "stage_1_ckpt_path: ${stage_1_ckpt_path}"
echo "datasetpath: ${datasetpath}"
rm ${ckpt_dir}/${exp} -r
mkdir -p ${ckpt_dir}/${exp}/log_rest/
cp ./config/train_MAE.yaml ${ckpt_dir}/${exp}/log_rest/train_${now}.yaml
deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 11451 finetuning/ft_MAE.py \
    --stage 2 \
    --cfg ./config/train_MAE.yaml \
    --data_path ${datasetpath} \
    --vision_config ${vision_config_path} \
    --delta_ckpt_path ${stage_1_ckpt_path} \
    --device cuda \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
    
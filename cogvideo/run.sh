#!/bin/bash

NLAYERS=48
NHIDDEN=3072
NATT=48
MAXSEQLEN=1024
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MPSIZE=1

#SAMPLING ARGS
TEMP=1.05
TOPK=12
export CUDA_VISIBLE_DEVICES=0

#script_path=$(realpath $0)
#script_dir=$(dirname $script_path)
#root_dir=$(dirname $script_dir)
#root_dir=$(dirname $script_path)
root_dir=$HOME/cogvideo
ds_config_path=$root_dir/scripts
seed=1234

MASTER_PORT=${MASTER_PORT} SAT_HOME=$root_dir/cogview-new python $root_dir/cogvideo_pipeline.py \
        --input-source $root_dir/datas/fetv_prompt_cn.txt \
        --output-path $root_dir/output_videos \
        --parallel-size 1 \
        --both-stages \
        --use-guidance-stage1 \
        --guidance-alpha 3.0 \
        --generate-frame-num 5 \
        --tokenizer-type fake \
        --mode inference \
        --distributed-backend nccl \
        --fp16 \
        --model-parallel-size $MPSIZE \
        --temperature $TEMP \
        --coglm-temperature2 0.89 \
        --top_k $TOPK \
        --sandwich-ln \
        --seed $seed \
        --num-workers 0 \
        --batch-size 1 \
        --max-inference-batch-size 1 \
	--deepspeed_config $ds_config_path/ds_config_compress.json \
        $@

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

OUTPUT_DIR="../datas/videos/zeroscope_feats"
LOG_DIR="../datas/videos/zeroscope_feats/logs"
PARTITION='video'
NNODE=1
NUM_GPUS=1
NUM_CPU=112


CUDA_VISIBLE_DEVICES=3 torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29200 \
    tasks/video_feature_extract.py \
    $(dirname $0)/configs_fvd/zeroscope_config.py \
    pretrained_path ckpts/umt_msrvtt_7k/ckpt_best.pth \
    batch_size 16 \
    output_dir ${OUTPUT_DIR}

seeds=(12 22 32 42)
model=cogvideo

for ind in 0 1 2 3
do
	seed=${seeds[$ind]}
	echo "Computing FVD for seed$seed"
	CUDA_VISIBLE_DEVICES=2 python src/scripts/calc_metrics_for_dataset.py \
		--real_data_path ../datas/videos/real_videos_fid/16frames_offset \
		--fake_data_path ../datas/videos/${model}_fid/16frames_offset \
		--save_path ../auto_eval_results/fvd_results/$model \
		--metrics fvd32_16f,fvd64_16f,fvd128_16f,fvd256_16f,fvd300_16f,fvd512_16f,fvd1024_16f \
		--mirror 1 \
		--gpus 1 \
		--resolution 256 \
		--verbose 0 \
		--use_cache 0 \
		--seed $seed
done

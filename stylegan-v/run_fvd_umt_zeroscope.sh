seeds=(12 22 32 42)
model=zeroscope

for ind in 0 1 2 3
do
	seed=${seeds[$ind]}
	echo "Computing FVD for seed$seed"
	CUDA_VISIBLE_DEVICES=0 python src/scripts/calc_metrics_for_dataset.py \
		--real_feat_path ../datas/videos/real_videos_feats/real_videos_feats.pt \
		--fake_feat_path ../datas/videos/${model}_feats/${model}_feats.pt \
		--save_path ../auto_eval_results/fvd_umt_results/${model} \
		--metrics fvd32_16f,fvd64_16f,fvd128_16f,fvd256_16f,fvd300_16f,fvd512_16f,fvd1024_16f \
		--mirror 1 \
		--gpus 1 \
		--resolution 256 \
		--verbose 0 \
		--use_cache 0 \
		--seed $seed
done

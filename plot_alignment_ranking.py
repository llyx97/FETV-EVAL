# plot the auto and human ranking of models in terms of tv alignment

import matplotlib.pyplot as plt
import numpy as np
from auto_human_correlation import load_multi_human_results, load_multi_eval_results, summarize_results

models = ['cogvideo', 'modelscope-t2v', 'text2video-zero', 'zeroscope', 'ground-truth']
model_name = {'cogvideo': 'CogVideo', 'modelscope-t2v': 'ModelScope-T2V', 'text2video-zero': 'Text2Video-zero', 'zeroscope': 'ZeroScope', 'ground-truth': 'Ground-Truth'}
metrics = ['CLIPScore', 'CLIPScore-ft', 'BLIPScore', 'Otter-VQA', 'UMTScore']


manual_results_raw = load_multi_human_results(['manual_eval_results/human0', 'manual_eval_results/human1', 'manual_eval_results/human2'], models)
manual_results = {}
for model in manual_results_raw:
    rut, ruch, ruo, idut, iduch, iduo, results_per_sent_id = summarize_results(eval_results=manual_results_raw[model], metrics=['alignment'],
                                                                                         data_file="datas/fetv_data.json")
    results_ = {k: v['alignment'] for k,v  in ruch.items()}
    results_.update({'all': np.mean(list(results_per_sent_id['alignment'].values()))})
    manual_results[model] = results_
print(manual_results['cogvideo'])

auto_eval_results = {}
for metric in metrics:
    scores = load_multi_eval_results(root_path=f'auto_eval_results/{metric}', model_names=models, prefix='auto_eval_results')

    avg_scores = {model: {'all': np.mean(list(scores[model].values()))} for model in scores}
    for model in models:
        for challenge, sent_ids in iduch.items():
            avg_scores[model][challenge] = np.mean([scores[model][sid] for sid in sent_ids if sid in scores[model]])
    auto_eval_results[metric] = avg_scores

# print(auto_eval_results)

for catgry in manual_results['cogvideo']:
    if catgry=='none':
        continue
    fig, axs = plt.subplots(figsize=(4*len(metrics), 4), nrows=1, ncols=len(metrics))
    manual_result = [manual_results[model][catgry] for model in models]
    for ax, metric in zip(axs, metrics):
        auto_result = [auto_eval_results[metric][model][catgry] for model in models]
        for model in models:
            ax.scatter(manual_results[model][catgry], auto_eval_results[metric][model][catgry], label=model_name[model], s=150)
        ax.set_ylabel(metric+r'$\uparrow$', fontsize=20)
        ax.set_xlabel(r'Human$\uparrow$', fontsize=20)
        ax.tick_params(labelsize=15)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), loc='lower center', ncol=5, fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    plt.show()
    # plt.savefig(f"tv_alignment_rank_correlation_{catgry.replace(' ', '_')}.pdf", format="pdf")
    plt.savefig(f"tv_alignment_rank_correlation_{catgry.replace(' ', '_')}.png", format="png")
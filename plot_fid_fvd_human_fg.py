# Plot fid, fvd and human evaluation of static, temporal quality
import matplotlib.pyplot as plt
import numpy as np
import json
from auto_human_correlation import load_multi_human_results, summarize_results

seeds = [12, 22, 32, 42]
models = ['cogvideo', 'modelscope-t2v', 'text2video-zero', 'zeroscope']
model_name = {'cogvideo': 'CogVideo', 'modelscope-t2v': 'ModelScopeT2V', 'text2video-zero': 'Text2Video-zero', 'zeroscope': 'ZeroScope'}
metrics = ['fid', 'fvd', 'fvd_umt']
metric_names = {'fid': r'FID$\downarrow$', 'fvd': 'FVD$\downarrow$', 'fvd_umt': 'FVD-UMT$\downarrow$'}

def load_result(fn):
    with open(fn, 'r') as f:
        result = json.load(f)
    if 'objects' in result:
        result['spatial'] = result['objects']
        del result['objects']
    return result


results = {}

manual_results = load_multi_human_results(['manual_eval_results/human0', 'manual_eval_results/human1', 'manual_eval_results/human2'], models)

manual_results_aspect = {'spatial': {}, 'temporal': {}}
for model in manual_results:
    rut, ruch, ruo, idut, iduch, iduo, results_per_sent_id = summarize_results(eval_results=manual_results[model], metrics=['static_quality', 'temporal_quality'],
                                                                                         data_file="datas/fetv_data.json")
    manual_results_aspect['spatial'][model] = ruo
    manual_results_aspect['temporal'][model] = rut

for aspect in ['spatial', 'temporal']:
    results[aspect] = {}
    for model in manual_results_aspect[aspect]:
        # Organize manual eval results
        for catgry in manual_results_aspect[aspect][model]:
            if not catgry in results[aspect]:
                results[aspect][catgry] = {}
            for metric in ['static_quality', 'temporal_quality']:
                if not metric in results[aspect][catgry]:
                    results[aspect][catgry][metric] = {}
                results[aspect][catgry][metric][model] = manual_results_aspect[aspect][model][catgry][metric]
            if not 'overall_quality' in results[aspect][catgry]:
                results[aspect][catgry]['overall_quality'] = {}
            results[aspect][catgry]['overall_quality'][model] = np.mean([results[aspect][catgry][metric][model] for metric in ['static_quality', 'temporal_quality']])
        # Organize auto eval results
        for metric in metrics:
            for seed in seeds:
                auto_results = load_result(fn=f'auto_eval_results/{metric}_fg_results/{model}/{seed}.json')
                for catgry in auto_results[aspect]:
                    if not metric in results[aspect][catgry]:
                        results[aspect][catgry][metric] = {}
                    if not model in results[aspect][catgry][metric]:
                        results[aspect][catgry][metric][model] = [auto_results[aspect][catgry]]
                    else:
                        results[aspect][catgry][metric][model] += [auto_results[aspect][catgry]]
            for catgry in results[aspect]:
                for model in results[aspect][catgry][metric]:
                    results[aspect][catgry][metric][model] = np.mean(results[aspect][catgry][metric][model])


for aspect in results:
    for catgry in results[aspect]:
        results_ = results[aspect][catgry]
        fig, axs = plt.subplots(figsize=(5*len(metrics), 4), nrows=1, ncols=len(metrics))
        for ax, metric in zip(axs, metrics):
            for model in models:
                ax.scatter(results_['overall_quality'][model], results_[metric][model], label=model_name[model], s=200)
            ax.set_ylabel(metric_names[metric], fontsize=20)
            ax.set_xlabel(r'Human$\uparrow$', fontsize=20)
            ax.tick_params(labelsize=15)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), loc='lower center', ncol=4, fontsize=15)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        plt.show()
        plt.savefig(f"fid_fvd_correlation_{aspect}_{catgry.replace(' ', '_')}.png", format="png")
        # plt.savefig(f"fid_fvd_correlation_{aspect}_{catgry.replace(' ', '_')}.pdf", format="pdf")
# Plot fid, fvd and human evaluation of static, temporal quality
import matplotlib.pyplot as plt
import numpy as np
import json
from auto_human_correlation import load_multi_human_results

seeds = [12, 22, 32, 42]
models = ['cogvideo', 'modelscope-t2v', 'text2video-zero', 'zeroscope']
model_name = {'cogvideo': 'CogVideo', 'modelscope-t2v': 'ModelScopeT2V', 'text2video-zero': 'Text2Video-zero', 'zeroscope': 'ZeroScope'}
metrics = ['fid', 'fvd', 'fvd_umt']
metric_names = {'fid': r'FID$\downarrow$', 'fvd': 'FVD$\downarrow$', 'fvd_umt': 'FVD-UMT$\downarrow$'}

def load_result(fn, metric):
    try:
        with open(fn, 'r') as f:
            result = json.load(f)['results'][metric]
    except json.decoder.JSONDecodeError:
        with open(fn, 'r') as f:
            lines = f.readlines()
            result = json.loads(lines[0])['results'][metric]
    return result


results = {}
for metric in metrics:
    results[metric] = {}
    for model in models:
        m = metric.replace('_umt', '')
        results[metric][model] = np.mean([load_result(fn=f'auto_eval_results/{metric}_results/{model}/{seed}/metric-{m}1024_16f.jsonl', metric=f'{m}1024_16f') for seed in seeds])

manual_results = load_multi_human_results(['manual_eval_results/human0', 'manual_eval_results/human1', 'manual_eval_results/human2'], models)

for metric in ['static_quality', 'temporal_quality', 'alignment']:
    results[metric] = {}
    for model in models:
        results[metric][model] = np.mean([r[metric] for r in manual_results[model].values()])
results['overall_quality'] = {model: np.mean([results['static_quality'][model], results['temporal_quality'][model]]) for model in models}
print(results)

fig, axs = plt.subplots(figsize=(15, 4), nrows=1, ncols=3)
for ax, metric in zip(axs, ['fid', 'fvd', 'fvd_umt']):
    for model in models:
        ax.scatter(results['overall_quality'][model], results[metric][model], label=model_name[model], s=200)
    ax.set_ylabel(metric_names[metric], fontsize=20)
    ax.set_xlabel(r'Human$\uparrow$', fontsize=20)
    ax.tick_params(labelsize=15)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), loc='lower center', ncol=4, fontsize=15)
plt.tight_layout()
fig.subplots_adjust(bottom=0.25)
plt.show()
plt.savefig(f"fid_fvd_correlation.png", format="png")
# plt.savefig(f"fid_fvd_correlation.pdf", format="pdf")
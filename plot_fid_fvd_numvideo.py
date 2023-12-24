# Plot fid results with different number of video samples

import matplotlib.pyplot as plt
import numpy as np
import json

seeds = [12, 22, 32, 42]
models = ['cogvideo', 'modelscope-t2v', 'text2video-zero', 'zeroscope']
model_name = {'cogvideo': 'CogVideo', 'modelscope-t2v': 'ModelScopeT2V', 'text2video-zero': 'Text2Video-zero', 'zeroscope': 'ZeroScope'}
num_videos = [64, 128, 256, 300, 512, 1024]
metrics = {m: {num: f'{m}{num}_16f' for num in num_videos} for m in ['fid', 'fvd']}
metrics['fvd_umt'] = {num: f'fvd{num}_16f' for num in num_videos}
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

fig, axs = plt.subplots(figsize=(15, 4), nrows=1, ncols=3)

for ax, metric in zip(axs, metrics):
    for model in models:
        results = {'mean': [],'std': [], 'max': [], 'min': []}
        for num in metrics[metric]:
            raw_results = [load_result(fn=f'auto_eval_results/{metric}_results/{model}/{seed}/metric-{metrics[metric][num]}.jsonl', metric=metrics[metric][num]) for seed in seeds]
            results['mean'].append(np.mean(raw_results))
            results['std'].append(np.std(raw_results))
            results['max'].append(np.max(raw_results))
            results['min'].append(np.min(raw_results))
        ax.plot(num_videos, results['mean'], 'o--', label=model_name[model])
        ax.fill_between(num_videos, results['min'], results['max'], alpha=0.2)
        ax.set_ylabel(metric_names[metric], fontsize=20)
        ax.set_xlabel('# Videos', fontsize=20)
    ax.tick_params(labelsize=15)
    ax.axvline(x=300, color='grey')
    ax.set_xticks([100, 300, 500, 800, 1000], ['100', '300', '500', '800', '1000'])
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), loc='lower center', ncol=4, fontsize=15)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
plt.show()
plt.savefig(f"fid_fvd_numvideo.png", format="png")
# plt.savefig(f"fid_fvd_numvideo.pdf", format="pdf")
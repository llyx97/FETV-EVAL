import json, os, argparse
import numpy as np

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


s_contents = ['animals', 'people' , 'plants', 'illustrations', 'artifacts', 'vehicles', 'buildings & infrastructure', 
               'food & beverage', 'scenery & natural objects']
t_contents = ['fluid motions', 'light change', 'actions', 'kinetic motions']
challenges = {
    'complexity': ['simple', 'medium', 'complex'],
    'spatial': ['color', 'quantity', 'camera view'],
    'temporal': ['speed', 'motion direction', 'event order'],
}


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def flatten(nest_list:list):
    return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]


def collect_sent_id(text_dir, limit=10000):
    filename = text_dir
    ids_under_spatial = {key: [] for key in s_contents}
    ids_under_temporal = {key: [] for key in t_contents}
    ids_under_challenge = {key: [] for key in flatten(list(challenges.values()))+['none']}  # none denotes no specific attribute to control
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    lines = lines[:limit]
    for sent_id, line in enumerate(lines):
        data = json.loads(line)
        assert all([obj in s_contents for obj in data['major content']['spatial']])
        for obj in data['major content']['spatial']:
            ids_under_spatial[obj].append(str(sent_id))

        if data['major content']['temporal'] is not None:
            assert all([t in t_contents for t in data['major content']['temporal']])
            for t in data['major content']['temporal']:
                ids_under_temporal[t].append(str(sent_id))

        challenges_ = flatten(list(data['attribute control'].values()))
        challenges_ = list(set(challenges_))
        if None in challenges_:
            challenges_.remove(None)
        if len(challenges_)==0:
            challenges_= ['none']
        challenges_ += data['prompt complexity']
        for c in challenges_:
            ids_under_challenge[c].append(str(sent_id))
    return ids_under_temporal, ids_under_challenge, ids_under_spatial

def find_latest_eval_result(prefix, root_path='.'):
    eval_files = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.startswith(prefix) and f.endswith('.json')]
    if len(eval_files)==0:
        return
    latest_file = max(eval_files, key=os.path.getctime)
    return latest_file


def summarize_results(eval_results, data_file, metrics, limit=10000):   
    """
        Summarize results under different categories, given manual eval results of a single model
    """
    # ids_under_temporal, ids_under_challenge, ids_under_spatial
    idut, iduch, idus = collect_sent_id(data_file, limit=limit)
    results_per_sent_id = {metric: {sid: result[metric] for sid, result in eval_results.items() if int(sid)<limit and metric in result} for metric in metrics}

    # results_under_temporal, results_under_challenge, results_under_spatial
    rut, ruch, rus = {}, {}, {}
    for temporal, sent_ids_ in idut.items():
        rut[temporal] = {metric: np.mean([results_[sid] for sid in sent_ids_ if sid in results_]) for metric, results_ in results_per_sent_id.items()}
        rut[temporal]["Num_sample"] = len(sent_ids_)
    for obj, sent_ids_ in idus.items():
        rus[obj] = {metric: np.mean([results_[sid] for sid in sent_ids_ if sid in results_]) for metric, results_ in results_per_sent_id.items()}
        rus[obj]["Num_sample"] = len(sent_ids_)
    for challenge, sent_ids_ in iduch.items():
        ruch[challenge] = {metric: np.mean([results_[sid] for sid in sent_ids_ if sid in results_]) for metric, results_ in results_per_sent_id.items()}
        ruch[challenge]["Num_sample"] = len(sent_ids_)
        # fine-grained_alignment
        if challenge in ['color', 'quantity', 'camera view', 'speed', 'motion direction', 'event order']:
            fg_alignment_score = np.mean([result['fine-grained_alignment'][challenge] for result in eval_results.values() if ('fine-grained_alignment' in result and challenge in result['fine-grained_alignment'])])
            ruch[challenge].update({'fine-grained_alignment': fg_alignment_score})
    return rut, ruch, rus, idut, iduch, idus, results_per_sent_id

def load_single_eval_results(result_path, limit=10000): # load the evaluation results of a single t2v models
    eval_results = {}
    with open(result_path, 'r') as f:
        lines = f.readlines()[:limit]
        for line in lines:
            eval_results.update(json.loads(line))
    # sort the eval_results according to the data id
    if 'manual_eval' in result_path:
        eval_results = {int(id): value for id, value in eval_results.items()}
        eval_results = dict(sorted(eval_results.items()))
        eval_results = {str(id): value for id, value in eval_results.items()}
    return eval_results

def load_multi_eval_results(model_names=['cogvideo', 'text2video-zero', 'damo-text2video'], root_path='manual_eval_results', prefix='manual_eval_results'): # load the evaluation results of a multiple t2v models, from a single human evaluator
    eval_results = {}
    for m_name in model_names:
        result_path = find_latest_eval_result(prefix=f"{prefix}_{m_name}", root_path=root_path)
        eval_results[m_name] = load_single_eval_results(result_path)
    return eval_results

def load_multi_human_results(human_paths, model_names): # Load evaluation results from multiple human evaluators
    manual_results = {}
    for i, human_path in enumerate(human_paths):
        manual_results[f'human{i}'] = load_multi_eval_results(root_path=human_path, model_names=model_names)

    manual_results['avg'] = {model: {} for model in model_names}
    for model in model_names:
        for id in manual_results[f'human0'][model]:
            manual_results['avg'][model][id] = {}
            for key in manual_results[f'human0'][model][id]:
                if key=='video_id':
                    manual_results['avg'][model][id][key] = manual_results[f'human0'][model][id][key]
                elif key=='fine-grained_alignment':
                    manual_results['avg'][model][id][key] = {}
                    for dim in manual_results['human0'][model][id][key]:
                        avg_score = np.mean([manual_results[f'human{i}'][model][id][key][dim] for i in range(len(human_paths))])
                        manual_results['avg'][model][id][key][dim] = avg_score
                else:
                    avg_score = np.mean([manual_results[f'human{i}'][model][id][key] for i in range(len(human_paths))])
                    manual_results['avg'][model][id][key] = avg_score
    return manual_results['avg']


def radar_plot(eval_results, aspects, model_names):
    """
        Args:
            eval_results: evaluation results of multiple t2v models
            aspects: different aspects of categories
            model_names: the name of t2v models
    """
    
    catry_name_map = {'fluid motions': 'Fluid Motions', 'light change': 'Light\nChange', 
                      'kinetic motions': 'Kinetic\nMotions', 'actions': 'Action',
                'buildings & infrastructure': 'Building', 'food & beverage': 'Food', 'scenery & natural objects': 'Scenery', 'plants': 'Plants',
                'animals': 'Animals', 'people': 'People', 'vehicles': 'Vehicles', 'artifacts': 'Artifacts', 'illustrations': 'Illustrations',
                'motion direction': 'Motion\nDirection', 'camera view': 'Camera\nView', 'event order': 'Event\nOrder', 'speed': 'Speed', 'color': 'Color', 'quantity': 'Quantity'}
    model_name_map = {'damo-text2video': 'ModelScopeT2V', 'text2video-zero': 'Text2Video-zero', 'cogvideo': 'CogVideo', 'ground-truth': 'Ground-Truth', 'zeroscope': "ZeroScope"}
    fig_size = {'Temporal': (12, 6.3), 'Spatial': (12, 6.3), 'Controlability': (12, 6.7), 'Complexity': (6,5.8)}

    colors = ['paleturquoise', 'firebrick', 'tab:purple','green', 'yellow']
    any_model_name = list(eval_results.keys())[0]
    for aspect in aspects:

        assert aspect in ['Temporal', 'Spatial', 'Complexity', 'Controlability']

        any_rux = eval_results[any_model_name][aspect]
        theta = radar_factory(len(any_rux), frame='circle')
        if aspect in ['Temporal', 'Spatial']:
            metrics = ['static_quality', 'temporal_quality']
        elif aspect=='Controlability':
            metrics = ['alignment', 'fine-grained_alignment']
        elif aspect=='Complexity':
            metrics = ['alignment']
        fig, axs = plt.subplots(figsize=fig_size[aspect], nrows=1, ncols=len(metrics),
                        subplot_kw=dict(projection='radar'))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

        for ax, metric in zip(axs, metrics):
            ax.set_rgrids([1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5])
            if metric=='temporal_quality':
                ax.set_rgrids([1.0, 2.0, 3.0, 4.5])
            if aspect=='Complexity':
                ax.set_rgrids([3.0, 3.5, 4.0, 4.5])

            metric_name = 'overall alignment' if metric=='alignment' else metric
            ax.set_title(metric_name.replace('_', ' '), weight='bold', size=30, position=(0.5, 1.5),
                        horizontalalignment='center', verticalalignment='center')

            for mid, model_name in enumerate(model_names):
                rux = eval_results[model_name][aspect]
                scores = [rux[key][metric] for key in rux]
                spoke_labels = list(rux.keys())
                for i, label in enumerate(spoke_labels):
                    if label in catry_name_map:
                        spoke_labels[i] = catry_name_map[label]
                ax.plot(theta, scores, color=colors[mid], label=model_name_map[model_name], zorder=0, linewidth=7)
                ax.fill(theta, scores, facecolor=colors[mid], alpha=0.05, label='_nolegend_')
                ax.set_varlabels(spoke_labels)
            ax.tick_params(labelsize=20, axis='x', zorder=10)
            ax.tick_params(labelsize=16, axis='y')

        # Add a legend for the whole figure.
        handles, labels = axs[0].get_legend_handles_labels()
        ncol = 2 if aspect=='Complexity' else 5
        if not aspect=='Complexity':
            leg = fig.legend(handles, labels, loc='lower center', ncol=ncol, fontsize=15)
            for line in leg.get_lines():
                line.set_linewidth(12.0)
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"Figures/manual_result_{aspect}.png", format="png")
        # plt.savefig(f"Figures/manual_result_{aspect}.pdf", format="pdf")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()      
    parser.add_argument('--model_names', nargs='+', default=['ground-truth', 'damo-text2video', 'cogvideo', 'text2video-zero', 'zeroscope'])        
    parser.add_argument('--manual_result_paths', nargs='+', default=['manual_eval_results/human0', 'manual_eval_results/human1', 'manual_eval_results/human2'], help="manual_result_paths of different humans")  
    parser.add_argument('--metrics', nargs='+', default=['static_quality', 'temporal_quality', 'alignment'])
    args = parser.parse_args()

    manual_results = load_multi_human_results(args.manual_result_paths, args.model_names)

    manual_results_aspect = {}    # manual_results under each category or challenge aspect
    for model_name in manual_results:
        rut, ruch, rus, idut, iduch, idus, results_per_sent_id = summarize_results(eval_results=manual_results[model_name], metrics=args.metrics,
                                                                                         data_file="fetv_data.json")
        for key in results_per_sent_id:
            avg_score = np.mean(list(results_per_sent_id[key].values()))
            print(model_name, key, f"{avg_score:.2f}")   # Print overall result of all t2v models
        ru_complx = {key: ruch[key] for key in ['simple', 'medium', 'complex']}  # results under complexity
        ru_control = {key: ruch[key] for key in ['color', 'quantity', 'camera view', 'speed', 'motion direction', 'event order']}  # results under controlability
        manual_results_aspect[model_name] = {'Complexity': ru_complx, 'Controlability': ru_control, 'Temporal': rut, 'Spatial': rus}

    # Radar plot of fine-grained results
    radar_plot(manual_results_aspect, 
               aspects=['Temporal', 'Spatial', 'Complexity', 'Controlability'],
               model_names=args.model_names)
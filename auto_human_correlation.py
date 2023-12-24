import cv2, json, os, re, datetime, argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.metrics import cohen_kappa_score

import tkinter as tk
from tkinter import filedialog
import os
import cv2
from PIL import Image, ImageTk
import pandas as pd
import krippendorff
from scipy.stats import spearmanr

objects = ['animals', 'people' , 'plants', 'illustrations', 'artifacts', 'vehicles', 'buildings & infrastructure', 
               'food & beverage', 'scenery & natural objects']
temporal = ['fluid motions', 'light change', 'actions', 'kinetic motions']
challenges = {
    'complexity': ['simple', 'medium', 'complex'],
    'attribute': ['color', 'quantity', 'camera view'],
    'temporal': ['speed', 'motion direction', 'event order'],
}


def flatten(nest_list:list):
    return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]

def print_result(rux, exclusions=None):
        """
            exclusions: list of categories or challenges not to print.
        """
        max_str_len = max([len(key) for key in rux])
        for i, (key, results) in enumerate(rux.items()):
            if i==0:
                line = [myAlign(key, 15) for key in results]
                print(myAlign('Metrics', max_str_len+5) + ''.join(line))
            if exclusions is not None and key in exclusions:
                continue
            line = ""
            for metric, score in results.items():
                if metric!='Num_sample':
                    line += myAlign('%.3f'%score, 15)
                else:
                    line += myAlign('%d'%score, 15)
            print(myAlign(key, max_str_len+5) + line)
        print("-"*100)

def myAlign(string, length=0):
    if length == 0:
        return string
    slen = len(string)
    re = string
    if isinstance(string, str):
        placeholder = ' '
    else:
        placeholder = u'　'
    while slen < length:
        re += placeholder
        slen += 1
    return re


def collect_sent_id(text_dir, limit=10000):
    filename = text_dir
    ids_under_objects = {key: [] for key in objects}
    ids_under_temporal = {key: [] for key in temporal}
    ids_under_challenge = {key: [] for key in flatten(list(challenges.values()))}
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    lines = lines[:limit]
    for sent_id, line in enumerate(lines):
        data = json.loads(line)
        assert all([obj in objects for obj in data['major content']['spatial']])
        for obj in data['major content']['spatial']:
            ids_under_objects[obj].append(str(sent_id))

        if data['major content']['temporal'] is not None:
            assert all([t in temporal for t in data['major content']['temporal']])
            for t in data['major content']['temporal'] :
                ids_under_temporal[t].append(str(sent_id))

        challenges_ = flatten(list(data['attribute control'].values()) + data['prompt complexity'])
        challenges_ = list(set(challenges_))
        if None in challenges_:
            challenges_.remove(None)
        challenges_ = set(challenges_).intersection(flatten(list(challenges.values())))
        for c in challenges_:
            ids_under_challenge[c].append(str(sent_id))
    return ids_under_temporal, ids_under_challenge, ids_under_objects

def find_latest_eval_result(prefix, root_path='.'):
    eval_files = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.startswith(prefix) and f.endswith('.json')]
    if len(eval_files)==0:
        return
    latest_file = max(eval_files, key=os.path.getctime)
    return latest_file


def auto_correlation(auto_result_path, manual_result_path=None, raw_manual_results=None, overall_corr=False, model_names=['cogvideo'], metrics=['static_quality'], coeff='spearman', auto_metric=None):
    """
        analyze the auto-human correlation
        Args
            overall_corr: whether only compute overall correlation
            manual_result_path: the path of manual evaluation results from a single human, if set to None, the manual_results should be provided
    """
    correlations = {}
    auto_eval_results = load_multi_eval_results(root_path=auto_result_path, model_names=model_names, prefix='auto_eval_results')
    auto_eval_results = {auto_metric: {model_name: auto_eval_results[model_name] for model_name in model_names}}

    if manual_result_path is not None:
        raw_manual_results = load_multi_eval_results(root_path=manual_result_path, model_names=model_names)
    else:
        assert raw_manual_results is not None
    manual_eval_results = {}
    for model_name in raw_manual_results:
        rut, ruch, ruo, idut, iduch, iduo, results_per_sent_id = summarize_results(eval_results=raw_manual_results[model_name], metrics=metrics,
                                                                                        data_file="datas/fetv_data.json")
        manual_eval_results[model_name] = results_per_sent_id['alignment']

    for metric in auto_eval_results:
        if overall_corr:
            auto_result = flatten([list(auto_eval_results[metric][model_name].values()) for model_name in model_names])
            manual_result = flatten([list(manual_eval_results[model_name].values()) for model_name in model_names])
            auto_result = np.array(auto_result)
            manual_result = np.array(manual_result)
            kendall_corr, _, _, spear_coef, _ = compute_correlation(manual_result, auto_result, compute_cohen=False)
            correlations[metric] = spear_coef if coeff=='spearman' else kendall_corr['Kendall-c']
        else:
            correlations[metric] = {}
            for cont_dim in iduch:
                auto_result = flatten([[s for id,s in auto_eval_results[metric][model_name].items() if id in iduch[cont_dim]] for model_name in model_names])
                manual_result = flatten([[s for id,s in manual_eval_results[model_name].items() if id in iduch[cont_dim]] for model_name in model_names])
                auto_result = np.array(auto_result)
                manual_result = np.array(manual_result)
                kendall_corr, _, _, spear_coef, _ = compute_correlation(manual_result, auto_result, compute_cohen=False)
                correlations[metric][cont_dim] = spear_coef if coeff=='spearman' else kendall_corr['Kendall-c']

    return correlations


def summarize_results(eval_results, data_file, metrics, limit=10000):
    """
        Summarize results under different categories and challenges types, given manual eval results of a single model
    """
    # ids_under_category, ids_under_temporal, ids_under_challenge, ids_under_objects
    idut, iduch, iduo = collect_sent_id(data_file, limit=limit)
    results_per_sent_id = {metric: {sid: result[metric] for sid, result in eval_results.items() if int(sid)<limit and metric in result} for metric in metrics}

    # results_under_category, results_under_temporal and results_under_challenge
    rut, ruch, ruo = {}, {}, {}
    for temporal, sent_ids_ in idut.items():
        rut[temporal] = {metric: np.mean([results_[sid] for sid in sent_ids_ if sid in results_]) for metric, results_ in results_per_sent_id.items()}
        rut[temporal]["Num_sample"] = len(sent_ids_)
    for obj, sent_ids_ in iduo.items():
        ruo[obj] = {metric: np.mean([results_[sid] for sid in sent_ids_ if sid in results_]) for metric, results_ in results_per_sent_id.items()}
        ruo[obj]["Num_sample"] = len(sent_ids_)
    for challenge, sent_ids_ in iduch.items():
        ruch[challenge] = {metric: np.mean([results_[sid] for sid in sent_ids_ if sid in results_]) for metric, results_ in results_per_sent_id.items()}
        ruch[challenge]["Num_sample"] = len(sent_ids_)
        # fine-grained_alignment
        if challenge in ['color', 'quantity', 'camera view', 'speed', 'motion direction', 'event order']:
            fg_alignment_score = np.mean([result['fine-grained_alignment'][challenge] for result in eval_results.values() if ('fine-grained_alignment' in result and challenge in result['fine-grained_alignment'])])
            ruch[challenge].update({'fine-grained_alignment': fg_alignment_score})
    return rut, ruch, ruo, idut, iduch, iduo, results_per_sent_id

def load_single_eval_results(result_path, limit=10000):
    """
        load the evaluation results of a single t2v models
    """
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

def load_multi_eval_results(model_names=['cogvideo', 'text2video-zero', 'damo-text2video'], root_path='manual_eval_results', prefix='manual_eval_results'):
    """
        load the evaluation results of a multiple t2v models, from a single human evaluator
    """
    eval_results = {}
    for m_name in model_names:
        result_path = find_latest_eval_result(prefix=f"{prefix}_{m_name}", root_path=root_path)
        eval_results[m_name] = load_single_eval_results(result_path)
    return eval_results

def load_multi_human_results(human_paths, model_names):
    """
        Load evaluation results from multiple human evaluators
    """
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

def compute_correlation(auto_results, manual_results, compute_cohen=True):
    corr_kendall = {}
    for variant in ['c', 'b']:
        tau = ss.kendalltau(ss.rankdata(auto_results), ss.rankdata(manual_results), variant=variant)
        corr_kendall[f"Kendall-{variant}"] = tau.correlation

    if compute_cohen:
        corr_cohen = cohen_kappa_score(auto_results, manual_results)
    else:
        corr_cohen = 0
    
    df = pd.DataFrame({'auto_eval': auto_results, 'manual_eval': manual_results})
    kripp_alpha = krippendorff.alpha(reliability_data=df.values, level_of_measurement='nominal')

    spearman_coef, p_value = spearmanr(auto_results, manual_results)
    return corr_kendall, corr_cohen, kripp_alpha, spearman_coef, p_value


if "__main__" == __name__:
    parser = argparse.ArgumentParser()      
    parser.add_argument('--model_names', nargs='+', default=['cogvideo', 'text2video-zero', 'modelscope-t2v', 'ground-truth', 'zeroscope'])        
    parser.add_argument('--manual_result_paths', nargs='+', default=['manual_eval_results/human0', 'manual_eval_results/human1', 'manual_eval_results/human2'], help="manual_result_paths of different humans")  
    parser.add_argument('--metrics', nargs='+', default=['static_quality', 'temporal_quality', 'alignment'])
    parser.add_argument('--auto_result_paths', nargs='+', default=['auto_eval_results/CLIPScore', 'auto_eval_results/CLIPScore-ft', 'auto_eval_results/BLIPScore', 'auto_eval_results/Otter-VQA', 'auto_eval_results/UMTScore'])        
    args = parser.parse_args()
    

    # Analyze Auto-human Correlation
    manual_results = load_multi_human_results(args.manual_result_paths, args.model_names)
    correlations = {}
    for coeff in ['kendall', 'spearman']:
        print(f"{coeff} Coefficient:")
        for auto_path in args.auto_result_paths:
            auto_metric = auto_path.split('/')[1]
            corrs = auto_correlation(auto_path, raw_manual_results=manual_results, model_names=args.model_names, metrics=args.metrics, coeff=coeff, auto_metric=auto_metric)
            avg_corr = auto_correlation(auto_path, raw_manual_results=manual_results, model_names=args.model_names, metrics=args.metrics, overall_corr=True, coeff=coeff, auto_metric=auto_metric)
            for metric in avg_corr:
                corrs[metric]['Avg'] = avg_corr[metric]
            correlations.update(corrs)
        print_result(correlations)
import torch, os, argparse, json
import numpy as np
from model import Model

def read_prompts(prompt_path):
    prompts = []
    with open(prompt_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            data = json.loads(line)
            prompts.append(data["prompt"])
    return prompts

def inference_single_prompt(model, prompt, pid, output_root_path, fps=4, num_frame=8, model_path=None, seed=-1):
    params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "model_name": model_path,
                "video_length": num_frame, "seed": seed, "watermark": None}

    out_file = f"{output_root_path}/{fps}fps_{num_frame}frames/{pid}_{prompt.replace(' ','_')}.mp4"
    out_path, _ = os.path.split(out_file)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("--------------------------------------------")
    print(f"Generating video for {pid} prompt: {prompt}")
    print("--------------------------------------------")
    model.process_text2video(prompt, fps = fps, path = out_file, **params)

def inference_dataset(args):
    # prompts = ["A horse galloping on a street"]
    prompts = read_prompts(args.prompt_path)
    model = Model(device = "cuda", dtype = torch.float16)
    for pid, prompt in enumerate(prompts):
        inference_single_prompt(model, prompt, pid+586, args.output_root_path, args.fps, args.num_frame, args.model_path, args.seed)

if "__main__" == __name__:
    # config
    parser = argparse.ArgumentParser()     
    parser.add_argument('--output_root_path', default=None)     
    parser.add_argument('--model_path', default='sd_models/dreamlike-art')     
    parser.add_argument('--prompt_path', default=None)     
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_frame', default=16, type=int)
    parser.add_argument('--fps', default=8, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        _ = torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    inference_dataset(args)
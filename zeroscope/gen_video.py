import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import numpy as np
import argparse, json, os

def read_prompts(prompt_path):
    prompts = []
    with open(prompt_path, 'r') as fp:
        lines = fp.readlines()
        for id, line in enumerate(lines):
            data = json.loads(line)
            if 'prompt' in data:
                prompts.append({'id': id, 'prompt': data["prompt"]})
            elif 'sentences' in data:
                prompts.append({'id': id, 'prompt': data["sentences"]})
    return prompts

if "__main__" == __name__:
    # config
    # The number of frames is specified in weights/configuration.json
    parser = argparse.ArgumentParser()     
    parser.add_argument('--output_root_path', default='output_videos')     
    parser.add_argument('--prompt_path', default=None)     
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    _ = torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained("zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompts = read_prompts(args.prompt_path)
    if not os.path.exists(args.output_root_path):
        os.makedirs(args.output_root_path)

    for prompt in prompts:
        video_file = os.path.join(args.output_root_path, f"{prompt['id']}.mp4")
        if os.path.isfile(video_file):
            continue
        video_frames = pipe(prompt['prompt'], num_inference_steps=40, height=320, width=576, num_frames=24).frames
        video_path = export_to_video(video_frames, output_video_path=video_file)

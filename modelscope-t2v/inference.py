from huggingface_hub import snapshot_download

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib
import numpy as np
import torch, argparse, os, json


def read_prompts(prompt_path):
    prompts = []
    with open(prompt_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            data = json.loads(line)
            prompts.append(data["prompt"])
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

    if not os.path.exists(args.output_root_path):
        os.makedirs(args.output_root_path)

    model_dir = pathlib.Path('weights')
    # Uncomment the following two lines to download model checkpoints and related files
    # snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis',
    #                    repo_type='model', local_dir=model_dir)

    prompts = read_prompts(args.prompt_path)

    pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
    for pid, prompt in enumerate(prompts):
        test_text = {
                'text': prompt,
            }
        print("--------------------------------------------")
        print(f"Generating video for {pid} prompt: {prompt}")
        print("--------------------------------------------")
        output_path = f"{args.output_root_path}/{pid}_{prompt.replace(' ','_')}.mp4"
        output_video_path = pipe(test_text, output_video=output_path)[OutputKeys.OUTPUT_VIDEO]
        print('output_video_path:', output_video_path)
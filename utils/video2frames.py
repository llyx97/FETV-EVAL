import cv2, os, re, argparse
from tqdm import tqdm
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
import torchvision.transforms.functional as TVF

short_video_count = 0

def video2frame(
    video_path: os.PathLike, target_dir: os.PathLike, force_fps: int=None, num_frame: int=None,
    target_size: int=None, uniform_sampling: bool=False):
    """
      Args:
        num_frame: The number of frames to sample.
        uniform_sampling: Whether to use uniform sampling. If not, sample based of fps and num_frame.
    """

    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        print(f'Coudnt process video: {video_path}')
    os.makedirs(target_dir, exist_ok=True)

    if force_fps and not uniform_sampling:
      fps = force_fps
    else:
      fps = clip.fps
    # Ensure that the total number of frames is enough.
    global short_video_count
    if int(np.floor(clip.duration * fps)) < num_frame:
       short_video_count += 1
       print(f"Found {short_video_count} videos not long enough for the given fps and num_frame; Video length={clip.duration}, video fps={clip.fps}")
    while int(np.floor(clip.duration * fps)) < num_frame:
       fps += 1
    frames = [Image.fromarray(frame) for frame in clip.iter_frames(fps=fps)]

    if uniform_sampling:
       interval = max(len(frames)/num_frame, 1)
       frame_indices = np.arange(0, len(frames), interval, dtype=int)      # Uniformly sample num_frame indices
    else:
       start_frame = np.random.choice(np.arange(0, len(frames)-num_frame)) if len(frames)>num_frame else 0   # Randomly sample a starting frame index
       frame_indices = np.arange(start_frame, start_frame+num_frame, dtype=int)
    sampled_frames = [frame for i, frame in enumerate(frames) if i in frame_indices]
    for frame_idx, frame in enumerate(sampled_frames):
        if target_size is not None:
            frame = TVF.resize(frame, size=target_size, interpolation=Image.LANCZOS)
            frame = TVF.center_crop(frame, output_size=(target_size, target_size))
        frame.save(os.path.join(target_dir, f'frame{frame_idx}.jpg'), q=95)

def find_video_files(folder, ext='.mp4'):
    video_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(ext):
                video_files.append(os.path.join(root, file))
    return video_files

if "__main__" == __name__:
    parser = argparse.ArgumentParser()     
    parser.add_argument('--frm_num', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--force_fps', default=None, type=int, help="specifying the frame rate")
    parser.add_argument('--sample_num', default=-1, type=int, help='The maximum number of videos to process.')
    parser.add_argument('--video_ext', default='.mp4', type=str)
    parser.add_argument('--video_root_path', default='datas/videos/modelscope-t2v/videos', type=str)
    parser.add_argument('--target_root_path', default='datas/videos/modelscope-t2v', type=str)
    parser.add_argument('--sampling_strategy', default='uniform', type=str, choices=['uniform', 'offset'])
    args = parser.parse_args()

    np.random.seed(seed=args.seed)

    tgt_root = os.path.join(args.target_root_path, f"{args.frm_num}frames_{args.sampling_strategy}")
    if args.sampling_strategy=='offset':
        tgt_root = os.path.join(tgt_root+f"_{args.force_fps}fps", f"seed{args.seed}")

    video_files = find_video_files(folder=args.video_root_path, ext=args.video_ext)
    for i, vf in enumerate(tqdm(video_files)):
        sent_id = os.path.basename(vf).split('_')[0] if '_' in os.path.basename(vf) else os.path.basename(vf).split('.')[0]
        tgt_path = os.path.join(tgt_root, f'sent{sent_id}_frames')
        if not os.path.isdir(tgt_path):
            video2frame(video_path=vf, target_dir=tgt_path, num_frame=args.frm_num, 
                        force_fps=args.force_fps, uniform_sampling=args.sampling_strategy=='uniform')
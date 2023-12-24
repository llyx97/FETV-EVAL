import cv2, torchvision, torch, itertools, os, math, json, logging, sys
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from icetk import icetk

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

def read_video_cv2(video_path, save_img_directory, save_img_filename, frame_interval=5):
    """
        video_path: 视频文件所在的路径
        save_img_directory: 保存的图像文件所在的目录
        save_img_filename: 保存的图像文件的名字
        frame_interval: 隔多少帧存储一次图片 ，1表示连续帧
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total)
    pbar = tqdm(total= int(total/frame_interval))
    c=0                             #文件名从0开始
    while(1):
        # get a frame
        ret, frame = cap.read()
        if ret:
            #cv2.imshow("capture", frame)
            if c % frame_interval == 0:
                if c==0:
                    print(frame)
                    print(frame.shape)
                cv2.imwrite(save_img_directory + save_img_filename + str(c) +".jpg",frame) #存储为图像
                pbar.update(1)

        else:
            break
        c=c+1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def read_video_torchvision(video_object, start=0, end=None, read_video=True, read_audio=False):
    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            f"start time={start} and end time={end}"
        )

    video_frames = torch.empty(0)
    video_pts = []
    if read_video:
        video_object.set_current_stream("video")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] < end, video_object.seek(start)):
            frames.append(frame['data'])
            video_pts.append(frame['pts'])
        if len(frames) > 0:
            video_frames = torch.stack(frames, 0)

    audio_frames = torch.empty(0)
    audio_pts = []
    if read_audio:
        video_object.set_current_stream("audio")
        frames = []
        for frame in itertools.takewhile(lambda x: x['pts'] < end, video_object.seek(start)):
            frames.append(frame['data'])
            audio_pts.append(frame['pts'])
        if len(frames) > 0:
            audio_frames = torch.cat(frames, 0)

    return video_frames, audio_frames, (video_pts, audio_pts), video_object.get_metadata()

def video2image(video_path, save_path, framerate=1, frame_num=5):
    """
        video_path: path that contains multiple videos
        save_path: path to save the image files
    """
    video_files = os.listdir(video_path)
    video_files = [os.path.join(video_path, f) for f in video_files]
    pbar = tqdm(total = len(video_files))

    for v_file in video_files:
        video = torchvision.io.VideoReader(v_file, 'video')
        video_metadata = video.get_metadata()
        sample_interval = math.ceil(video_metadata['video']['fps'][0] / framerate) #sample one frame from every sample_interval frames
        video_id = v_file.strip('mp4').split('video')[-1].strip('.')

        single_clip_duration = (1/framerate)*frame_num
        video_clip_start, video_clip_end = 0, single_clip_duration
        while video_clip_end <= video_metadata['video']['duration'][0]:
            vf, af, info, meta = read_video_torchvision(video, video_clip_start, video_clip_end)
            saved_frame_num = 0
            for i, frame in enumerate(vf):
                if i%sample_interval == 0:
                    save_path_ = os.path.join(save_path, 'framerate%d'%framerate, video_id,
                            'clip%d_%d'%(video_clip_start, video_clip_end))
                    if not os.path.exists(save_path_):
                        os.makedirs(save_path_)
                    save_image(frame.float(), os.path.join(save_path_, f'{str(i).rjust(4,"0")}.jpg'), normalize=True)
                    saved_frame_num += 1
            if saved_frame_num != 5:
                print(video_id, saved_frame_num)
                print(video_metadata)
                print(sample_interval)
                print(video_clip_start, video_clip_end)
                print(len(vf))
            video_clip_start += single_clip_duration
            video_clip_end += single_clip_duration
        pbar.update(1)

def read_video_annotations(path):
    video_annotations = {}
    with open(annotation_path, 'r') as fp:
        sentences = json.load(fp)['sentences']
        for sent in sentences:
            video_id = sent['video_id']
            if video_id in video_annotations:
                video_annotations[video_id].append(sent['caption'])
            else:
                video_annotations[video_id] = [sent['caption']]
    return video_annotations

def encode_and_save_text_video(texts, img_root_path, output_file, max_text_len=64, duration=4.0):
    #Encode texts
    enc_texts, enc_videos = {}, {}
    for video_id in texts:
        enc_texts[video_id[-4:]] = [icetk.encode(t) for t in texts[video_id]]

    ####### Debug ########
    #a = set([int(i) for i in video_ids])
    #b = []
    #for v in texts:
    #    b.append(int(v[-4:]))
    #print(len(b), len(a))
    #print(len(set(b).intersection(a)))
    ###### End block ####

    #Encode video frames
    video_ids = os.listdir(img_root_path)
    pbar = tqdm(total = len(video_ids))
    logging.info("Encoding videos...")
    for video_id in video_ids:
        video_path = os.path.join(img_root_path, video_id)
        enc_videos[video_id] = []
        for clip_dir in os.listdir(video_path):
            video_tokens = []
            clip_path = os.path.join(video_path, clip_dir)
            img_files = os.listdir(clip_path)
            img_files.sort() # sort the image files according to the frame id in the video
            for img_file in img_files:
                img_tokens = icetk.encode(image_path=os.path.join(clip_path, img_file), 
                        image_size=160, compress_rate=8)  #TODO: check encoded image size
                video_tokens.append(img_tokens)
            video_tokens = torch.cat(video_tokens, dim=0)
            enc_videos[video_id].append(video_tokens.cpu())
        pbar.update(1)

    logging.info("Creating text-video pairs...")
    pbar = tqdm(total = len(video_ids))
    enc_tv_pairs = []
    for video_id in enc_videos:
        enc_ts = enc_texts[video_id]   # Every video_id correpond to multiple textual captions,
        enc_vs = enc_videos[video_id]  # and video clips.
        for enc_t in enc_ts:
            enc_t = icetk.encode(str(float(duration))+"秒<n>") + enc_t
            enc_t = enc_t[:min(max_text_len, len(enc_t))]
            enc_t_ = icetk['<pad>']*np.ones(max_text_len).astype(np.int32) #initialize as a sequence of '<pad>'
            enc_t_[:len(enc_t)] = enc_t
            for enc_v in enc_vs:
                pair = np.concatenate(
                        [np.array(enc_t_), enc_v.view(-1).numpy().astype(np.int32)]
                        )
                enc_tv_pairs.append(pair)
        pbar.update(1)

    enc_tv_pairs = np.array(enc_tv_pairs)
    logging.info("%d text-video pairs collected"%len(enc_tv_pairs))
    logging.info("Saving dataset to %s"%output_file)
    # Saving the np appay using np.memmap
    fp = np.memmap(output_file, dtype=enc_tv_pairs.dtype, mode='w+', shape=enc_tv_pairs.shape)
    fp[:] = enc_tv_pairs[:]
    del fp

video_path = "/home/videodata/MSRVTT/TestVideo"
img_save_path = "/home/liuyuanxin/CogVideo/datasets/"
annotation_path = "/home/videodata/MSRVTT/test_videodatainfo.json"
#read_video_cv2(video_path, save_img_directory, save_img_filename)

#video2image(video_path, img_save_path)
texts = read_video_annotations(annotation_path)
encode_and_save_text_video(texts, output_file='/home/liuyuanxin/CogVideo/datasets/stage1_data.npy', 
        img_root_path='/home/liuyuanxin/CogVideo/datasets/framerate1')

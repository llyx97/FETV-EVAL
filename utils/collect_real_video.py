import json, os, shutil, requests
import pandas as pd

msrvtt_src_path = '/path/to/downloaded/msrvtt/videos'
webvid_info_file = 'webvid_info_val.csv'

def mycopyfile(srcfile, dstpath):                       
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       
        shutil.copy(srcfile, os.path.join(dstpath, fname))          
        print ("copy %s -> %s"%(srcfile, os.path.join(dstpath, fname)))

def download(url, output_file):
    req = requests.get(url)
    filename = output_file
    if req.status_code != 200:
        print('Download error')
        return
    try:
        with open(filename, 'wb') as f:
            f.write(req.content)
            print(f'Download successfully to {output_file}')
    except Exception as e:
        print(e)


if "__main__" == __name__:
    video_id_file = "datas/fetv_data.json"
    with open(video_id_file, 'r') as f:
        lines = f.readlines()
    if video_id_file.endswith('.txt'):
        video_ids = [l.strip() for l in lines]
    elif video_id_file.endswith('.json'):
        video_ids = [str(json.loads(line)['video_id']) for line in lines]

    tgt_path = "datas/videos/real_videos" if "fetv_data" in video_id_file else "datas/videos/real_videos_fid"

    # Read Webvid Urls
    data = pd.read_csv(webvid_info_file, sep=',')
    url_map = {str(ind): url for ind, url in zip(data['videoid'], data['contentUrl'])}

    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)

    for video_id in video_ids:
        if os.path.isfile(os.path.join(tgt_path, video_id+'.mp4')):
            continue
        # Collect MSRVTT videos from local
        if 'video' in video_id:
            v_file = os.path.join(msrvtt_src_path, video_id+'.mp4')
            mycopyfile(v_file, tgt_path)
            continue
        # Download WebVid videos
        else:
            output_file = os.path.join(tgt_path, video_id+'.mp4')
            print(output_file)
            download(url_map[video_id], output_file)

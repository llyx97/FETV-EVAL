import json

datas_real, datas_gen_mp4, datas_gen_gif = [], [], []  # real videos and generated videos
with open('fetv_data.json', 'r') as f:
    lines = f.readlines()
    for id, line in enumerate(lines):
        d = json.loads(line)
        datas_gen_mp4.append({"caption_id": id, "video": f'{id}.mp4', "caption": d['prompt']})
        datas_gen_gif.append({"caption_id": id, "video": f'{id}.gif', "caption": d['prompt']})
        if d['video_id'] is not None:
            datas_real.append({"caption_id": id, "video": f"{d['video_id']}.mp4", "caption": d['prompt']})

with open('fetv_anno_real.json', 'w') as f:
    json.dump(datas_real, f)
with open('fetv_anno_gen_mp4.json', 'w') as f:
    json.dump(datas_gen_mp4, f)
with open('fetv_anno_gen_gif.json', 'w') as f:
    json.dump(datas_gen_gif, f)
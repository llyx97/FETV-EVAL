from moviepy.video.io.VideoFileClip import VideoFileClip
import os


def clip_video(src_file, tgt_file, start=0, end=1000):
    assert src_file.endswith('.mp4') and tgt_file.endswith('.mp4')
    tgt_dir = os.path.dirname(tgt_file)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    clip = VideoFileClip(src_file).subclip(start, end)
    clip.write_videofile(tgt_file)


clip_info = {
    'video7087': [0,2],
    'video9511': [5,10],
    'video9103': [8,12],
    'video7542': [0.6,4],
    'video9263': [11,15],
    '27254191': [47,54],
    'video9377': [6.5,14],
    'video7912': [5,9],
    'video9637': [4,8],
    'video8101': [20,27],
    'video8114': [5,10],
    'video7201': [7,10],
    'video9136': [0,9],
    '15680626': [12,14],
    'video9622': [8,14],
    'video9460': [3,7],
    'video7931': [3,5.2],
    'video9957': [0,7],
    'video7606': [2.5,8],
    'video9167': [3,8],
    'video8369': [3.5,5.5],
    'video8099': [8,13],
    'video9194': [5.5,9],
    'video7872': [4,10],
    '26733235': [0,6],
    'video9396': [14,26],
    'video7285': [13,19],
    'video9172': [10,17],
    'video9604': [0.8,7],
    'video7657': [4,13],
    'video8645': [7.5,11],
    'video7258': [6,13],
    'video9618': [0,3.5],
    'video9148': [0,2],
    'video9446': [0,0.7],
    'video9939': [0,3],
    'video8190': [12,19],
    'video8692': [8.3,11],
    'video8665': [0,8],
    'video8867': [9,17],
    'video8106': [9,28],
    'video7060': [1.4,3.9],
    'video7436': [6,8.5],
    'video9132': [7.5,11],
    'video7084': [1,4],
    'video9100': [0.2,1],
    'video9214': [7,12],
    'video7958': [20.2,24],
    'video8372': [6,9],
    'video7566': [0, 3],
    'video9461': [9,11],
    'video7623': [4.4,7.4],
    'video8138': [6,18],
    'video8179': [0,1],
    'video8871': [4,9],
    'video7100': [0,5],
    'video9321': [0,2.9],
    'video9481': [4.6,10],
    'video7137': [0,3],
}

for src_file, start_end in clip_info.items():
    src_file = src_file+'.mp4'
    tgt_file = src_file
    if 'video9511' in src_file:
        tgt_file = 'video9511_1.mp4'  # this video is used twice with different clips
    src_file = os.path.join('real_videos', src_file)
    tgt_file = os.path.join('real_videos', tgt_file)
    start, end = start_end[0], start_end[1]
    clip_video(src_file, tgt_file, start, end)
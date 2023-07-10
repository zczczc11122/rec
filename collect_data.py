import os
root = os.getcwd()
import csv
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
import requests
import subprocess
import librosa
import json

img_path = '/opt/tiger/mlx_notebook/image/'
wav_path = '/opt/tiger/mlx_notebook/wav/'
audio_path = '/opt/tiger/mlx_notebook/audio/'

def download(entry):
    object_id, uri = entry
    path = os.path.join(img_path, '{}.mp4'.format(object_id))
    wavpath = os.path.join(wav_path, '{}.wav'.format(object_id))
    audiopath_ = os.path.join(audio_path, '{}.npy'.format(object_id))
    if not os.path.exists(img_path + str(object_id)) or len(os.listdir(img_path + str(object_id))) == 0 or not os.path.exists(audiopath_):
        print(f'download {object_id}.mp4 to {path}...')
        #if str(object_id) not in dict_need:
        #    return path
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
            try:
                path2 = img_path + str(object_id)
                if not os.path.exists(path2):
                    cmd = 'mkdir' + path2
                    path2 += '/'
                    a2 = subprocess.Popen(cmd, shell=True)
                    out, err = a2.communicate()
                cmd = '../ffmpeg-git-20210724-amd64-static/ffmpeg -loglevel panic -t 60 -i ' + path + ' -f wav -ac 1 -ar 32000 ' + wavpath
                print(cmd)
                a2 = subprocess.Popen(cmd, shell=True)
                out, err = a2.communicate()
                if len(os.listdir(img_path + str(object_id))) == 0:
                    cmd = '../ffmpeg-git-20210724-amd64-static/ffmpeg -t 60 -i ' + path + ' -r 1 -f image2 {}%05d.jpeg'.format(path2)
                    print(cmd)
                    a2 = subprocess.Popen(cmd, shell=True)
                    out, err = a2.communicate()
            except:
                print('fail download!', path)
            os.remove(path)
    return wavpath, img_path + str(object_id)

def float32_to_int64(x):
    """float32 转化 int16 压缩存储空间"""
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)

def Convert_wav2npy(vid, wav_path):
    """
    下载音频wav文件
    input: 视频vid , 视频wav文件路径
    output: npy文件路径
    """
    # 每帧采样率
    sample_rate = 32000
    # 帧长 10s
    if os.path.exists(os.path.join(audio_path, vid + '.npy')):
        return None
    if os.path.isfile(wav_path):
        (audio, _) = librosa.core.load(wav_path, sr=sample_rate, mono=True)
        audio = float32_to_int64(audio)
        np.save(os.path.join(audio_path, vid + '.npy'), audio)
        os.remove(wav_path)
        return os.path.join(audio_path, vid + '.npy')
    else:
        print('no find file', wav_path)
        return None

def resize_image(img):
    h, w, _ = img.shape
    if max(h, w) < 800:
        return img
    if h > w:
        new_w = 528
        new_h = min(int(new_w * h / w // 32 * 32), 1280)
        img = cv2.resize(img, (new_w, new_h), cv2.INTER_CUBIC)
    else:
        new_h = 528
        new_w = min(int(new_h * w / h // 32 * 32), 1280)
        img = cv2.resize(img, (new_w, new_h), cv2.INTER_CUBIC)
    return img

def resize_path(imgpath):
    if not os.path.exists(imgpath):
        return None
    for image_name in os.listdir(imgpath):
        if 'jpeg' in image_name:
            image_file = os.path.join(imgpath, image_name)
            image = cv2.imread(image_file)
            image = resize_image(image)
            cv2.imwrite(image_file, image)

def main_ser(entry):
    print(entry)
    try:
        object_id, uri = entry
        wavpath, imgpath = download(entry)
        Convert_wav2npy(object_id, wavpath)
        resize_path(imgpath)
    except:
        pass


if __name__ == '__main__':

    csv_path = './dataset_ouraudit.csv'
    lines = csv.reader(open(csv_path, 'r'))
    entrys = []
    for line in lines:
        try:
            object_id = line[0].strip()
            #info = json.loads(line[1])
            entrys.append([object_id, line[1]])
        except Exception as e:
            print(e)
            continue
    print(len(entrys))
    print(entrys[0])
    p = ThreadPool(10)
    pool_output = p.map(main_ser, entrys)

import torch.utils.data as data
import pandas as pd
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pickle
import torch
from transformers import BertTokenizer

class VideoRecord(object):
    def __init__(self, vid, video_path, audio_path, title, ocr, num_frames, label):
        self._vid = vid
        self._video_path = video_path
        self._audio_path = audio_path
        self._title = title
        self._ocr = ocr
        self._num_frames = num_frames
        self._label = label

    @property
    def vid(self):
        return self._vid

    @property
    def video_path(self):
        return self._video_path

    @property
    def audio_path(self):
        return self._audio_path

    @property
    def title(self):
        return self._title

    @property
    def ocr(self):
        return self._ocr

    @property
    def num_frames(self):
        return int(self._num_frames)

    @property
    def label(self):
        return self._label

def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)

def pad_or_truncate(x, audio_length):
    """Pad 或者 切割音频文件到固定长度"""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[:audio_length]

class ListFileDataSet(data.Dataset):
    def __init__(self,
                 prefix_path,
                 info_file,
                 list_file,
                 label_dict,
                 num_segments,
                 dim="expression", #person | expression | style | topic
                 train=True,
                 image_tmpl='{:05d}.jpg',
                 transform=None,
                 bert_path='',
                 bert_max_len=50,
                 local_rank=-1
                 ):
        self.prefix_path = prefix_path
        self.info_file = info_file
        self.list_file = list_file
        self.label_dict = label_dict
        self.num_segments = num_segments
        self.dim = dim

        self.train = train
        self.image_tmpl = image_tmpl
        self.transform = transform

        self.bert_path = bert_path
        self.bert_max_len = bert_max_len

        self.local_rank = local_rank

        self._parse_info()
        self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        record = self.video_list[index]
        vid, feature = self._get(record)
        return vid, feature

    def _load_images(self, record, indices):
        images = []
        for idx in indices:
            # idx = idx + 1
            img_path = os.path.join(record.video_path, self.image_tmpl.format(int(idx)))
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        return images

    def _parse_list(self):
        self.video_list = []
        with open(self.list_file, "rb") as f:
            vid_list = pickle.load(f)
        id2cls = self.label_dict[self.dim]["id2cls"]
        cls2id = self.label_dict[self.dim]["cls2id"]
        for vid in vid_list:
            v_path = os.path.join(self.prefix_path, "frames", str(vid))
            if not os.path.exists(v_path):
                if self.local_rank in (0, -1):
                    print('img path is miss', v_path)
                continue
            num_frames = len(os.listdir(v_path))
            if num_frames <= 0:
                if self.local_rank in (0, -1):
                    print('img path is empty', v_path)
                continue
            a_path = os.path.join(self.prefix_path, "audio", str(vid) + ".npy")
            if not os.path.exists(a_path):
                if self.local_rank in (0, -1):
                    print('audio path is miss', a_path)
                continue
            label = self.vid2info[str(vid)][self.dim]
            if label not in cls2id:
                # print('img label is undefine', v_path, label)
                continue
            label_id = cls2id[label]
            title = self.vid2info[str(vid)]['title']
            ocr = self.vid2info[str(vid)]['ocr']
            video_record = VideoRecord(vid=str(vid),
                                       video_path=v_path,
                                       audio_path=a_path,
                                       title=title,
                                       ocr=ocr,
                                       num_frames=num_frames,
                                       label=label_id)

            self.video_list.append(video_record)

    def _parse_info(self):
        self.info_df = pd.read_parquet(self.info_file, engine='pyarrow').fillna("")
        self.vid2info = {}
        info_list = self.info_df.values.tolist()
        for i in info_list:
            vid, url, title, title_cut, ocr, ocr_cut, topic, topic_id, \
            style, style_id, expression, expression_id, person, person_id = i
            self.vid2info[str(vid)] = {}
            self.vid2info[str(vid)]['title'] = str(title)
            self.vid2info[str(vid)]['ocr'] = str(ocr)
            self.vid2info[str(vid)]['topic'] = str(topic)
            self.vid2info[str(vid)]['style'] = str(style)
            self.vid2info[str(vid)]['expression'] = str(expression)
            self.vid2info[str(vid)]['person'] = str(person)


    def _tsn_sample_indices(self, record):
        num_segments = self.num_segments
        frame_len = record.num_frames
        if self.train:
            average_duration = frame_len // num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration, size=num_segments)
            else:
                offsets = np.sort(randint(frame_len, size=(num_segments - frame_len)).tolist() + list(range(frame_len)))
            offsets = offsets.tolist()
        else:
            tick = frame_len / float(num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
            offsets = offsets.tolist()
        return offsets

    def _get(self, record):
        vid = record.vid

        indices = self._tsn_sample_indices(record)
        images = self._load_images(record, indices)

        images_q = self.transform(images)
        images_k = self.transform(images)

        return vid, (images_q, images_k)




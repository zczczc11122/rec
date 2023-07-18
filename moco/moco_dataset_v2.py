import torch.utils.data as data
import pandas as pd
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pickle


class VideoRecord(object):
    def __init__(self, vid, video_path, audio_path, title, ocr, num_frames,
                 expression_labels, material_labels, person_labels, style_labels, topic_labels):
        self._vid = vid
        self._video_path = video_path
        self._audio_path = audio_path
        self._title = title
        self._ocr = ocr
        self._num_frames = num_frames
        self._expression_labels = expression_labels
        self._material_labels = material_labels
        self._person_labels = person_labels
        self._style_labels = style_labels
        self._topic_labels = topic_labels

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
    def expression_labels(self):
        return self._expression_labels

    @property
    def material_labels(self):
        return self._material_labels

    @property
    def person_labels(self):
        return self._person_labels

    @property
    def style_labels(self):
        return self._style_labels

    @property
    def topic_labels(self):
        return self._topic_labels

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
                 train=True,
                 image_tmpl='{:03d}.jpg',
                 transform=None,
                 bert_path='',
                 bert_max_len=50,
                 local_rank=-1,
                 sep="|"
                 ):
        self.prefix_path = prefix_path
        self.info_file = info_file
        self.list_file = list_file
        self.label_dict = label_dict
        self.num_segments = num_segments

        self.train = train
        self.image_tmpl = image_tmpl
        self.transform = transform

        self.bert_path = bert_path
        self.bert_max_len = bert_max_len

        self.local_rank = local_rank
        self.sep = sep

        self._parse_info()
        self._parse_list()

        self.weights = None

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

    def _convert_label_2id(self, vid, dim):
        level_label = []
        label_str = self.vid2info[str(vid)][dim]
        labels = label_str.split(self.sep)
        label_map = self.label_dict[dim]
        max_level = len(label_map)
        for level in range(max_level):
            temp = self.sep.join(labels[0:level + 1])
            level_label.append(label_map[level]['cls2id'][temp])
        return level_label

    def _parse_list(self):
        self.video_list = []
        with open(self.list_file, "rb") as f:
            vid_list = pickle.load(f)
        for vid in vid_list:
            if str(vid) not in self.vid2info:
                continue
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
            a_path = os.path.join(self.prefix_path, "audio_npy", str(vid) + ".npy")
            if not os.path.exists(a_path):
                if self.local_rank in (0, -1):
                    print('audio path is miss', a_path)
                continue
            title = self.vid2info[str(vid)]['title']
            ocr = self.vid2info[str(vid)]['ocr']

            expression_label_ids = self._convert_label_2id(vid, 'expression')
            material_label_ids = self._convert_label_2id(vid, 'material')
            person_label_ids = self._convert_label_2id(vid, 'person')
            style_label_ids = self._convert_label_2id(vid, 'style')
            topic_label_ids = self._convert_label_2id(vid, 'topic')

            video_record = VideoRecord(vid=str(vid),
                                       video_path=v_path,
                                       audio_path=a_path,
                                       title=title,
                                       ocr=ocr,
                                       num_frames=num_frames,
                                       expression_labels=expression_label_ids,
                                       material_labels=material_label_ids,
                                       person_labels=person_label_ids,
                                       style_labels=style_label_ids,
                                       topic_labels=topic_label_ids)
            self.video_list.append(video_record)

    def _parse_info(self):
        self.info_df = pd.read_parquet(self.info_file, engine='pyarrow').fillna("")
        self.vid2info = {}
        info_list = self.info_df.values.tolist()
        for i in info_list:
            vid, video_url, frame_dir, audio_file, title, \
            title_cut, ocr, ocr_cut, copywriting, label_person, \
            label_scene, label_style, label_expression, label_material = i

            if len(label_expression) == 0:
                # print('label_expression is empty', vid)
                continue
            if len(label_material) == 0:
                # print('label_material is empty', vid)
                continue
            if len(label_person) == 0:
                # print('label_person is empty', vid)
                continue
            if len(label_style) == 0:
                # print('label_style is empty', vid)
                continue
            if len(label_scene) == 0:
                # print('label_scene is empty', vid)
                continue

            self.vid2info[str(vid)] = {}
            self.vid2info[str(vid)]['title'] = str(title)
            self.vid2info[str(vid)]['ocr'] = str(ocr)

            self.vid2info[str(vid)]['expression'] = str(label_expression[0]['name'])
            self.vid2info[str(vid)]['material'] = str(label_material[0]['name'])
            self.vid2info[str(vid)]['person'] = str(label_person[0]['name'])
            self.vid2info[str(vid)]['style'] = str(label_style[0]['name'])
            self.vid2info[str(vid)]['topic'] = str(label_scene[0]['name'])


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





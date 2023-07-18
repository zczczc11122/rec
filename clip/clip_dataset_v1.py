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
    def __init__(self, vid, video_path, text, num_frames, label):
        self._vid = vid
        self._video_path = video_path
        self._text = text
        self._num_frames = num_frames
        self._label = label

    @property
    def vid(self):
        return self._vid

    @property
    def video_path(self):
        return self._video_path

    @property
    def text(self):
        return self._text

    @property
    def num_frames(self):
        return int(self._num_frames)

    @property
    def label(self):
        return self._label

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
                 bert_max_len=79, # 对生成的文本数据进行统计，再设置
                 local_rank=-1,
                 sep=None
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
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.local_rank = local_rank

        self._parse_info()
        self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        record = self.video_list[index]
        vid, feature, label = self._get(record)
        return vid, feature, label

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

            label = self.vid2info[str(vid)][self.dim]
            if label not in cls2id:
                # print('img label is undefine', v_path, label)
                continue
            label_id = cls2id[label]
            title = self.vid2info[str(vid)]['title']
            en2_ch = {'expression': '表现形式', 'person': '人物', 'style': '风格', 'topic': '主题'}

            text = (f"视频{en2_ch[self.dim]}维度标签为{label}。" + "标题为" + title + "。").replace(" ", '').replace("\t", '，')

            video_record = VideoRecord(vid=str(vid),
                                       video_path=v_path,
                                       text=text,
                                       num_frames=num_frames,
                                       label=label_id)
            self.video_list.append(video_record)
        # text_l = {}
        # for i in self.video_list:
        #     print(i.text)
        #     if len(i.text) in text_l:
        #         text_l[len(i.text)] += 1
        #     else:
        #         text_l[len(i.text)] = 1
        # import pprint
        # print((sorted(text_l.items(), key=lambda x: x[0])))

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
        images = self.transform(images)

        text_text_idx = self.tokenizer.encode_plus(record.text,
                                                   max_length=self.bert_max_len,
                                                   padding='max_length',
                                                   truncation=True)
        text_input_ids = torch.LongTensor(text_text_idx['input_ids'])
        text_token_type_ids = torch.LongTensor(text_text_idx['token_type_ids'])
        text_attention_mask = torch.LongTensor(text_text_idx['attention_mask'])

        label = record.label
        return vid, (images, text_input_ids, text_token_type_ids, text_attention_mask), label



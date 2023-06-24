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
        return self._num_frames

    @property
    def label(self):
        return self._label

def float32_to_int64(x):
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
                 # pad_size=32,
                 # word_idx_file=None
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

        # self.pad_size = pad_size
        # with open(word_idx_file, 'rb') as fh:
        #     self.word_to_idx = pickle.load(fh)

        self._parse_info()
        self._parse_list()

        self.weights = self._get_sample_balanced_weights()

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
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        return images

    def _parse_list(self):
        self.video_list = []
        with open(self.list_file, 'rb') as f:
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
                offsets = np.sort(randint(frame_len, size=num_segments - frame_len)).tolist() + list(range(frame_len))
            offsets = offsets.tolist()
        else:
            tick = frame_len / float(num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
            offsets = offsets.tolist()
        return offsets

    def get_audio_tensor(self, record):
        audio_npy = np.load(record.audio_path)
        waveform = int16_to_float32(audio_npy)
        waveform = pad_or_truncate(waveform, 32000 * 10)
        waveform = waveform[::2]
        audio_tensor = torch.from_numpy(waveform).float()
        return audio_tensor

    # def _get_text_idx(self, record, text_type='title'):
    #     if text_type == 'title':
    #         pad_size = self.pad_size
    #     else:
    #         pad_size = self.pad_size * 2
    #     vocab = self.word_to_idx
    #     UNK, PAD = '<UNK>', '<PAD>'
    #     word_idx_list = []
    #     if text_type == 'title':
    #         word_cut = record.title_word_cut
    #     else:
    #         word_cut = record.ocr_word_cut
    #     for word in word_cut:
    #         if word in vocab:
    #             idx = vocab[word]
    #             word_idx_list.append(idx)
    #         else:
    #             for w in word:
    #                 word_idx_list.append(vocab.get(w, vocab.get(UNK)))
    #     if len(word_idx_list) < pad_size:
    #         pad_idx = vocab.get(PAD)
    #         word_idx_list.extend([pad_idx] * (pad_size - len(word_idx_list)))
    #     else:
    #         word_idx_list = word_idx_list[:pad_size]
    #     word_idx_list = torch.LongTensor(word_idx_list)
    #     return word_idx_list

    def _get(self, record):
        vid = record.vid

        indices = self._tsn_sample_indices(record)
        images = self._load_images(record, indices)
        images = self.transform(images)

        audio = self.get_audio_tensor(record)

        title_text_idx = self.tokenizer.encode_plus(record.title,
                                                    max_length=self.bert_max_len,
                                                    padding='max_length',
                                                    truncation=True)
        ocr_text_idx = self.tokenizer.encode_plus(record.ocr,
                                                  max_length=self.bert_max_len,
                                                  padding='max_length',
                                                  truncation=True)
        title_inputs_ids = torch.LongTensor(title_text_idx['input_ids'])
        title_token_type_ids = torch.LongTensor(title_text_idx['token_type_ids'])
        title_attention_mask = torch.LongTensor(title_text_idx['attention_mask'])
        ocr_inputs_ids = torch.LongTensor(ocr_text_idx['input_ids'])
        ocr_token_type_ids = torch.LongTensor(ocr_text_idx['token_type_ids'])
        ocr_attention_mask = torch.LongTensor(ocr_text_idx['attention_mask'])

        label = record.label

        # title_text_idx = self._get_text_idx(record, text_type='title')
        # ocr_text_idx = self._get_text_idx(record, text_type='ocr')
        # return vid, (images, title_text_idx, ocr_text_idx), label

        return vid, (images, audio, title_inputs_ids, title_token_type_ids, title_attention_mask,
                     ocr_inputs_ids, ocr_token_type_ids, ocr_attention_mask), label

    def _get_sample_balanced_weights(self):
        self.label2weight = {}
        for record in self.video_list:
            label = record.label
            if label not in self.label2weight:
                self.label2weight[label] = 0
            self.label2weight[label] += 1
        for label in self.label2weight:
            self.label2weight[label] = len(self.video_list) / float(self.label2weight[label])
        weights = []
        for record in self.video_list:
            label = record.label
            weights.append(self.label2weight[label])
        return weights



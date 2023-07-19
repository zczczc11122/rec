import torch.utils.data as data
import pandas as pd
from PIL import Image
import os
import os.path
import random
import numpy as np
from numpy.random import randint
import pickle
import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

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

class Txt_db:
    def __init__(self,
                 vocab_size,
                 mask,
                 cls,
                 sep):
        self.vocab_size = vocab_size
        self.mask = mask
        self.cls = cls
        self.sep = sep


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

def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label) and len(tokens) > 0:
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label

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
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.local_rank = local_rank
        self.sep = sep

        self.txt_db = Txt_db(self.tokenizer.vocab_size,
                             self.tokenizer.mask_token_id,
                             self.tokenizer.cls_token_id,
                             self.tokenizer.sep_token_id)


        self._parse_info()
        self._parse_list()

        self.weights = None

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        record = self.video_list[index]
        random_index = self.get_random_index(index)
        random_record = self.video_list[random_index]

        vid, images, audio, \
        title_input_ids, title_input_ids_mask, title_txt_labels_mask, title_token_type_ids_mask, title_attention_mask_mask, \
        ocr_input_ids, ocr_input_ids_mask, ocr_txt_labels_mask, ocr_token_type_ids_mask, ocr_attention_mask_mask, \
        random_images, itm_target, mrfr_img_mask, \
        expression_label_ids, material_label_ids, person_label_ids, style_label_ids, topic_label_ids = self._get(record, random_record)
        return vid, images, audio,\
               title_input_ids, title_input_ids_mask, title_txt_labels_mask, title_token_type_ids_mask, title_attention_mask_mask,\
               ocr_input_ids, ocr_input_ids_mask, ocr_txt_labels_mask, ocr_token_type_ids_mask, ocr_attention_mask_mask,\
               random_images, itm_target, mrfr_img_mask, \
               expression_label_ids, material_label_ids, person_label_ids, style_label_ids, topic_label_ids

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.vocab_size,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels

    def create_text_io(self, input_ids):
        input_ids = torch.tensor([self.txt_db.cls]
                                 + input_ids
                                 + [self.txt_db.sep])
        return input_ids

    def get_random_index(self, index):
        while True:
            random_index = random.randint(0, len(self.video_list) - 1)
            if random_index != index:
                break
        return random_index

    def _get_img_mask(self, mask_prob, num_bb):
        img_mask = [random.random() < mask_prob for _ in range(num_bb)]
        if not any(img_mask):
            # at least mask 1
            img_mask[random.choice(range(num_bb))] = True
        img_mask = torch.tensor(img_mask)
        return img_mask

    def _get_img_tgt_mask(self, img_mask, txt_len):
        z = torch.zeros(txt_len, dtype=torch.uint8)
        img_mask_tgt = torch.cat([z, img_mask], dim=0)
        return img_mask_tgt


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

    def get_audio_tensor(self, record):
        audio_npy = np.load(record.audio_path)
        waveform = int16_to_float32(audio_npy)
        waveform = pad_or_truncate(waveform, 32000 * 10)
        waveform = waveform[::2]
        audio_tensor = torch.from_numpy(waveform).float()
        return audio_tensor

    def _get(self, record, random_record):
        vid = record.vid
        itm_target = np.random.choice([0, 1], size=1, p=[0.5, 1 - 0.5])[0]

        indices = self._tsn_sample_indices(record)
        images = self._load_images(record, indices)
        images = self.transform(images)

        if itm_target == 0:
            random_indices = self._tsn_sample_indices(random_record)
            random_images = self._load_images(random_record, random_indices)
            random_images = self.transform(random_images)
        else:
            random_images = images

        audio = self.get_audio_tensor(record)

        title_res = self.tokenizer.tokenize(record.title)[:510]
        title_token_id = self.tokenizer.convert_tokens_to_ids(title_res)
        title_input_ids_mask, title_txt_labels_mask = self.create_mlm_io(title_token_id)
        title_input_ids_mask = torch.LongTensor(title_input_ids_mask)
        title_txt_labels_mask = torch.LongTensor(title_txt_labels_mask)
        title_token_type_ids_mask = torch.zeros(title_input_ids_mask.size(0), dtype=torch.long)
        title_attention_mask_mask = torch.ones(title_input_ids_mask.size(0), dtype=torch.long)

        title_input_ids = self.create_text_io(title_token_id)
        title_input_ids = torch.LongTensor(title_input_ids)

        ocr_res = self.tokenizer.tokenize(record.ocr)[:510]
        ocr_token_id = self.tokenizer.convert_tokens_to_ids(ocr_res)
        ocr_input_ids_mask, ocr_txt_labels_mask = self.create_mlm_io(ocr_token_id)
        ocr_input_ids_mask = torch.LongTensor(ocr_input_ids_mask)
        ocr_txt_labels_mask = torch.LongTensor(ocr_txt_labels_mask)
        ocr_token_type_ids_mask = torch.zeros(ocr_input_ids_mask.size(0), dtype=torch.long)
        ocr_attention_mask_mask = torch.ones(ocr_input_ids_mask.size(0), dtype=torch.long)

        ocr_input_ids = self.create_text_io(ocr_token_id)
        ocr_input_ids = torch.LongTensor(ocr_input_ids)

        num_segments = images.size()[0] // 3

        mrfr_img_mask = self._get_img_mask(0.15, num_segments)

        expression_label_ids = record.expression_labels
        material_label_ids = record.material_labels
        person_label_ids = record.person_labels
        style_label_ids = record.style_labels
        topic_label_ids = record.topic_labels

        return vid, images, audio,\
               title_input_ids, title_input_ids_mask, title_txt_labels_mask, title_token_type_ids_mask, title_attention_mask_mask,\
               ocr_input_ids, ocr_input_ids_mask, ocr_txt_labels_mask, ocr_token_type_ids_mask, ocr_attention_mask_mask,\
               random_images, itm_target, mrfr_img_mask, \
               expression_label_ids, material_label_ids, person_label_ids, style_label_ids, topic_label_ids



def data_collate(inputs):
    (vid, images, audio,
     title_input_ids, title_input_ids_mask, title_txt_labels_mask, title_token_type_ids_mask, title_attention_mask_mask,
     ocr_input_ids, ocr_input_ids_mask, ocr_txt_labels_mask, ocr_token_type_ids_mask, ocr_attention_mask_mask,
     random_images, itm_target, mrfr_img_mask,
     expression_label_ids,
     material_label_ids,
     person_label_ids,
     style_label_ids,
     topic_label_ids) = map(list, unzip(inputs))

    images = torch.stack(images, dim=0)
    audio = torch.stack(audio, dim=0)

    # title_lens_mask = [i.size(0) for i in title_input_ids_mask]
    title_input_ids = pad_sequence(title_input_ids, batch_first=True, padding_value=0)
    title_input_ids_mask = pad_sequence(title_input_ids_mask, batch_first=True, padding_value=0)
    title_txt_labels_mask = pad_sequence(title_txt_labels_mask, batch_first=True, padding_value=-1)
    title_token_type_ids_mask = pad_sequence(title_token_type_ids_mask, batch_first=True, padding_value=0)
    title_attention_mask_mask = pad_sequence(title_attention_mask_mask, batch_first=True, padding_value=0)

    # ocr_lens_mask = [i.size(0) for i in ocr_input_ids_mask]
    ocr_input_ids = pad_sequence(ocr_input_ids, batch_first=True, padding_value=0)
    ocr_input_ids_mask = pad_sequence(ocr_input_ids_mask, batch_first=True, padding_value=0)
    ocr_txt_labels_mask = pad_sequence(ocr_txt_labels_mask, batch_first=True, padding_value=-1)
    ocr_token_type_ids_mask = pad_sequence(ocr_token_type_ids_mask, batch_first=True, padding_value=0)
    ocr_attention_mask_mask = pad_sequence(ocr_attention_mask_mask, batch_first=True, padding_value=0)

    random_images = torch.stack(random_images, dim=0)
    itm_target = torch.tensor(itm_target)
    mrfr_img_mask = torch.stack(mrfr_img_mask, dim=0)

    expression_label_ids = torch.tensor(expression_label_ids).T
    material_label_ids = torch.tensor(material_label_ids).T
    person_label_ids = torch.tensor(person_label_ids).T
    style_label_ids = torch.tensor(style_label_ids).T
    topic_label_ids = torch.tensor(topic_label_ids).T

    batch = {'images': images,
             'audio': audio,
             'title_input_ids': title_input_ids,
             'title_input_ids_mask': title_input_ids_mask,
             'title_txt_labels_mask': title_txt_labels_mask,
             'title_token_type_ids_mask': title_token_type_ids_mask,
             'title_attention_mask_mask': title_attention_mask_mask,
             'ocr_input_ids': ocr_input_ids,
             'ocr_input_ids_mask': ocr_input_ids_mask,
             'ocr_txt_labels_mask': ocr_txt_labels_mask,
             'ocr_token_type_ids_mask': ocr_token_type_ids_mask,
             'ocr_attention_mask_mask': ocr_attention_mask_mask,
             'random_images': random_images,
             'itm_target': itm_target,
             'mrfr_img_mask': mrfr_img_mask,
             'expression_label_ids': expression_label_ids,
             'material_label_ids': material_label_ids,
             'person_label_ids': person_label_ids,
             'style_label_ids': style_label_ids,
             'topic_label_ids': topic_label_ids
             }
    return vid, batch


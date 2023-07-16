import os
import time
import torch
from transformers import BertTokenizer
run_num = 10
num_segments = 10
bert_path = ''
target_pt_path = ''

mytokenizer = BertTokenizer.from_pretrained(bert_path)
title_text_idx = mytokenizer.encode_plus('哈哈哈', max_length=50, padding='max_length', truncation=True)
title_input_ids = torch.LongTensor(title_text_idx['input_ids']).unsqueeze(0).cuda()
title_token_type_ids = torch.LongTensor(title_text_idx['token_type_ids']).unsqueeze(0).cuda()
title_attention_mask = torch.LongTensor(title_text_idx['attention_mask']).unsqueeze(0).cuda()

ocr_text_idx = mytokenizer.encode_plus('哈哈哈', max_length=50, padding='max_length', truncation=True)
ocr_input_ids = torch.LongTensor(ocr_text_idx['input_ids']).unsqueeze(0).cuda()
ocr_token_type_ids = torch.LongTensor(ocr_text_idx['token_type_ids']).unsqueeze(0).cuda()
ocr_attention_mask = torch.LongTensor(ocr_text_idx['attention_mask']).unsqueeze(0).cuda()

image = torch.rand(1, 3 * num_segments, 224, 224).cuda()
image2 = torch.rand(1, 3 * num_segments, 224, 224).cuda()
audio = torch.rand((1, 32000)).cuda()

features =[image, audio, title_input_ids, title_token_type_ids, title_attention_mask, ocr_input_ids,
           ocr_token_type_ids, ocr_attention_mask]

model_2 = torch.jit.load(os.path.join(target_pt_path, 'model.pt'))
output2 = model_2(*features)
start = time.time()
for i in range(run_num):
    output2 = model_2(*features)
end = time.time()
print(f'total: {end - start} run_num:{run_num}')
print('ave:', (end - start)/run_num)
print('fps:', run_num / (end - start))
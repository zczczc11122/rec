from transformers import AutoModel, AutoConfig, BertModel, BertConfig
import torch
import torch.nn as nn

class TextBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fintuing = args.bert_fintuing
        self.bert_path = args.bert_path
        self.config = BertConfig.from_pretrained(self.bert_path)
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.text_embedding_size = self.config.hidden_size
        # self.bert_path = '/opt/tiger/mlx_notebook/cc/xlm_roberta'
        # self.config = AutoConfig.from_pretrained(self.bert_path)
        # self.bert = AutoModel.from_pretrained(self.bert_path)
        # self.text_embedding_size = self.config.hidden_size

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        if not self.fintuing:
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask, token_type_ids)
                out_pool = outputs[1]
                return out_pool
        else:
            outputs = self.bert(input_ids, attention_mask, token_type_ids)
            out_pool = outputs[1]
            return out_pool
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(input_ids)
        # print(attention_mask)
        # if not self.fintuing:
        #     with torch.no_grad():
        #         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #         out_pool = outputs[1]
        #     return out_pool
        # else:
        #     outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #     out_pool = outputs[1]
        #     print(out_pool.shape)
        #     return out_pool

    def get_output_dim(self):
        return self.text_embedding_size


class TextBERT_seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fintuing = args.bert_fintuing
        self.bert_path = args.bert_path
        self.config = BertConfig.from_pretrained(self.bert_path)
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.text_embedding_size = self.config.hidden_size

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        if not self.fintuing:
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask, token_type_ids)
                out_pool = outputs[1]
                return out_pool
        else:
            outputs = self.bert(input_ids, attention_mask, token_type_ids)
            out_pool = outputs[1]
            return out_pool

    def get_output_dim(self):
        return self.text_embedding_size
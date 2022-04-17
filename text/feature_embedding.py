from transformers import AutoModel, AutoTokenizer
import torch
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import parser_helper as helper

config = helper.get_training_config()
MODEL_TYPE = config.pretrained_txt_model

tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
text_model = AutoModel.from_pretrained(MODEL_TYPE)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:2' if use_cuda else 'cpu')
print("Device: ", device)

text_model = text_model.to(device)

def extract_features(text, max_seq_len): #tokenizer, text_model):
    text = text.lower()

    encoded_input = tokenizer(text, padding="max_length", max_length=max_seq_len, return_tensors = "pt")

    encoded_input = encoded_input.to(device)

    with torch.no_grad():
        output = text_model(**encoded_input)

    semantic_feat = output.last_hidden_state
    # attention_mask = encoded_input['attention_mask']
    mask_s_idx = max_seq_len
    if 0 in encoded_input['input_ids'][0]:
        mask_s_idx = encoded_input['input_ids'].detach().cpu()[0].tolist().index(0)

    return semantic_feat.detach().cpu().numpy()[0], mask_s_idx





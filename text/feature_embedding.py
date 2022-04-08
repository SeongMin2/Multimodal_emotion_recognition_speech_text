from transformers import AutoModel, AutoTokenizer
import sys
import argparse
import parser_helper as helper

MODEL_NAME = "bert-base-uncased"
FEAT_TYPE = "last_hidden_state"

def extract_features(text, max_seq_len, tokenizer, text_model):
    text = text.lower()

    encoded_input = tokenizer(text, padding="max_length", max_length=max_seq_len, return_tensors = "pt")

    output = text_model(**encoded_input)

    semantic_feat = output.last_hidden_state


    return semantic_feat.detach().numpy()[0]





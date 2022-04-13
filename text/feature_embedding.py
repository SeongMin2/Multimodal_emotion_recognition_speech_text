def extract_features(text, max_seq_len, tokenizer, text_model):
    text = text.lower()

    encoded_input = tokenizer(text, padding="max_length", max_length=max_seq_len, return_tensors = "pt")

    output = text_model(**encoded_input)

    semantic_feat = output.last_hidden_state
    # attention_mask = encoded_input['attention_mask']
    mask_s_idx = max_seq_len
    if 0 in encoded_input['input_ids'][0]:
        mask_s_idx = encoded_input['input_ids'][0].tolist().index(0)


    return semantic_feat.detach().numpy()[0], mask_s_idx





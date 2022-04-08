from transformers import AutoModel, AutoTokenizer
import pandas as pd
import glob
from ABS_PATH import ABS_PATH

def get_length(tokens):
    return len(tokens)

def preprocess_text(text):
    text = text.lower()
    text = text.replace("\n", "")
    return text

ABS_PATH = ABS_PATH
txt_model = "bert-base-uncased"  
table_dir = ABS_PATH + "/full_data/table"

tables = glob.glob(table_dir + "/" + "*.csv")

tokenizer = AutoTokenizer.from_pretrained(txt_model)

# encoded_input = tokenizer(["Excuse me","my name is doctor"], padding="max_length", max_length=124, return_tensors = "pt")

max_token_length = 0

emotions = ["hap", "exc", "sad", "ang", "neu"]
txt_list = list()
for table_path in tables:
    df = pd.read_csv(table_path)

    for emotion in emotions:
        txt_list += df[df["emotion"] == emotion]["text"].values.tolist()

    txt_list = list(map(preprocess_text, txt_list))

    token_list = tokenizer(txt_list)

    token_lengths = list(map(get_length, token_list["input_ids"]))
    max_length = max(token_lengths)
    if max_length > max_token_length:
        max_token_length = max_length
    else:
        pass
    index = token_lengths.index(max_token_length)
    print(txt_list[index])
    print(tokenizer.convert_ids_to_tokens(token_list['input_ids'][index]))
    print(max_token_length)



# txt_model = "bert-base-uncased" 일 경우 tokenizer의 max size는 124임
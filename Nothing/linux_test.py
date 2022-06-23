import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__)).rsplit("/", 1)[0]
# linux에서는 path가 \ 아니라 / 로 잘 인식되어서 이방법 쓰면 안될 듯
print(ABS_PATH)

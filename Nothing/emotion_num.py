import pandas as pd
import glob
from ABS_PATH import ABS_PATH

ABS_PATH = ABS_PATH


def check_emotion_num():
    table_dir = ABS_PATH + "/full_data/table"
    tables = glob.glob(table_dir + "/" + "*.csv")

    hap_num = 0
    exc_num = 0
    sad_num = 0
    ang_num = 0
    neu_num = 0

    for table_path in tables:
        df = pd.read_csv(table_path)
        counts = df["emotion"].value_counts()
        hap_num += int(counts["hap"])
        exc_num += int(counts["exc"])
        ang_num += int(counts["ang"])
        sad_num += int(counts["sad"])
        neu_num += int(counts["neu"])

    total_num = hap_num + exc_num + sad_num + ang_num + neu_num
    print("hap: " + str(hap_num))
    print("exc: " + str(exc_num))
    print("ang: " + str(ang_num))
    print("sad: " + str(sad_num))
    print("neu: " + str(neu_num))
    print("total: " + str(total_num))

check_emotion_num()
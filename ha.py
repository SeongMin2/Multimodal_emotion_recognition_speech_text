import pandas as pd

data = {'file':['Ses01F_impro01_F000','Ses01F_impro01_M000'],'text':['Excuse me.','Do you have your forms?']}
df = pd.DataFrame(data)
df.to_csv('./all_sessions.csv',sep=';')
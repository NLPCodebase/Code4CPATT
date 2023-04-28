import pandas as pd


data = pd.read_csv("./data/event_train_eventstoryline.csv", sep=',',encoding = 'UTF-8').values.tolist()

# data = pd.read_csv("./data/event_eventstoryline_simple.csv", sep=',',encoding = 'UTF-8').values.tolist()
df = pd.DataFrame(data, columns=["Source sentence","Answer sentence","Event1","Event2","labels","type"])

# train_df=df[int(df.shape[0] * 0.2):]
# test_df=df[~df.index.isin(train_df.index)]
i = 0
test_df=df[int(df.shape[0] * 0.2 * i):int(df.shape[0] * 0.2 * (i + 1))]
train_df=df[~df.index.isin(test_df.index)]

train_outputpath = './data/train_eventstoryline.csv'
test_outputpath = './data/test_eventstoryline.csv'

 
train_df.to_csv(train_outputpath,sep=',',encoding = 'UTF-8',index=False,header=True) 
test_df.to_csv(test_outputpath,sep=',',encoding = 'UTF-8',index=False,header=True) 

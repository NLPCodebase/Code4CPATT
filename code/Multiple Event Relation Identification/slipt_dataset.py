import pandas as pd


data = pd.read_csv("./data/event_train.csv", sep=',',encoding = 'gbk').values.tolist()
df = pd.DataFrame(data, columns=["Source sentence","Answer sentence","Event1","Event2","labels"])
i = 4
test_df=df[int(df.shape[0] * 0.2 * i):int(df.shape[0] * 0.2 * (i + 1))]
train_df=df[~df.index.isin(test_df.index)]


train_outputpath = './data/train_multi.csv'
test_outputpath = './data/test_multi.csv'

 
train_df.to_csv(train_outputpath,sep=',',encoding = 'gbk',index=False,header=True) 
test_df.to_csv(test_outputpath,sep=',',encoding = 'gbk',index=False,header=True) 

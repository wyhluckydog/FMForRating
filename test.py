"""
这个文件是用来对数据进行处理，读取训练文件以及测试文件中的数据
"""
import pandas as pd
import torch as pt
from sklearn.feature_extraction import DictVectorizer

cols=['user','item','rating','timestamp']
train = pd.read_csv('data/ratings_train.csv', encoding='utf-8',names=cols)
test = pd.read_csv('data/ratings_test.csv', encoding='utf-8',names=cols)

train=train.drop(['timestamp'],axis=1)   #时间戳是不相关信息，可以去掉
test=test.drop(['timestamp'],axis=1)

# DictVectorizer会把数字识别为连续特征，这里把用户id和item id强制转为 catogorical identifier
train["item"]=train["item"].apply(lambda x:"c"+str(x))
train["user"]=train["user"].apply(lambda  x:"u"+str(x))

test["item"]=test["item"].apply(lambda x:"c"+str(x))
test["user"]=test["user"].apply(lambda x:"u"+str(x))

# 在构造特征向量时应该不考虑评分，只考虑用户数和电影数
train_no_rating=train.drop(['rating'],axis=1)
test_no_rating=test.drop(['rating'],axis=1)
all_df=pd.concat([train_no_rating,test_no_rating])
# all_df=pd.concat([train,test])
# data_num=all_df.shape
print("all_df shape",all_df.shape)
# 打印前10行
# print("all_df head",all_df.head(10))

# 进行特征向量化,有多少特征，就会新创建多少列
vec=DictVectorizer()
vec.fit_transform(all_df.to_dict(orient='record'))
# 合并训练集与验证集，是为了one hot,用完可以释放
del all_df

x_train=vec.transform(train.to_dict(orient='record')).toarray()
x_test=vec.transform(test.to_dict(orient='record')).toarray()
# print(vec.feature_names_)   #查看转换后的别名
print("x_train shape",x_train.shape)
print("x_test shape",x_test.shape)

y_train=train['rating'].values.reshape(-1,1)
y_test=test['rating'].values.reshape(-1,1)
print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)

n,p=x_train.shape

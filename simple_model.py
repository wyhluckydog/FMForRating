import pandas as pd
import torch as pt
from torch.autograd import Variable
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
data_num=all_df.shape
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

class FM_model(pt.nn.Module):
    def __init__(self,p,k):
        super(FM_model,self).__init__()
        self.p=p    #feature num
        self.k=k     #factor num
        self.linear=pt.nn.Linear(self.p,1,bias=True)   #linear part
        self.v=pt.nn.Parameter(pt.rand(self.k,self.p))  #interaction part

    def fm_layer(self,x):
        #linear part
        linear_part=self.linear(pt.tensor(x).float())
        #interaction part
        inter_part1=pt.mm(pt.tensor(x).float(),self.v.t())
        inter_part2=pt.mm(pt.pow(pt.tensor(x).float(),2),pt.pow(self.v,2).t())
        pair_interactions=pt.sum(pt.sub(pt.pow(inter_part1,2),inter_part2),dim=1)
        output=linear_part.transpose(1,0)+0.5*pair_interactions
        return output

    def forward(self, x):
        output=self.fm_layer(x)
        return output

k=10   #因子的数目
fm=FM_model(p,k)
fm
# print("paramaters len",len(list(fm.parameters())))
# # print(list(fm.parameters()))
# for name,param in fm.named_parameters():
#     if param.requires_grad:
#         print(name)

#训练网络
optimizer=pt.optim.SGD(fm.parameters(),lr=0.01)   #学习率为0.01
for i in range(20):
    optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
    output=fm(x_train)
    output=output.transpose(1,0)
    #平方差
    loss_func= pt.nn.MSELoss()
    mse_loss=loss_func(output,pt.tensor(y_train).float())
    l2_regularization=pt.tensor(0).float()
    # print("l2_regularization type",l2_regularization.dtype)
    #加入l2正则
    for param in fm.parameters():
        # print("param type",pt.norm(param,2).dtype)
        l2_regularization+=pt.norm(param,2)
    loss=mse_loss+l2_regularization
    loss.backward()
    optimizer.step()   #进行更新
    # print(loss)

# print("train_loss",loss)
# print(y_train[0:10],"  ",output[0:10])

#评价指标rmse
def rmse(pred_rate,real_rate):
    #使用均方根误差作为评价指标
    loss_func=pt.nn.MSELoss()
    mse_loss=loss_func(pred_rate,pt.tensor(real_rate).float())
    rmse_loss=pt.sqrt(mse_loss)
    return rmse_loss


#测试网络
pred=fm(x_test)
pred=pred.transpose(1,0)
rmse_loss=rmse(pred,y_test)
print("test_loss",rmse_loss)
print(y_test[0:5]," ",pred[0:5])

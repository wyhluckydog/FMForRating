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
    def __init__(self,n,p,k):
        super(FM_model, self).__init__()
        self.n=n   #测试用例的数目
        self.p=p
        self.k=k
        self.linear=pt.nn.Linear(self.n,1,bias=True)
        self.v=pt.rand((self.p,self.k),requires_grad=True)
        self.w=pt.rand((self.p,1),requires_grad=True)
        # self.v=pt.normal(0,0.01,(self.p,self.k))
        # self.w=pt.normal(0,0.1,(self.p,1))
        self.w0=pt.rand((self.n,1),requires_grad=True)

    def fm_layer(self,x):
        # print(self.w0[1:10])
        #FM算法的线性部分
        linear_term=pt.mm(pt.tensor(x).float(),self.w)
        # print("linear_term shape",linear_term.shape)
        pair_interactions=pt.sum(
            pt.sub(
                pt.pow(
                    pt.mm(pt.tensor(x).float(),self.v),
                    2
                ),
                pt.mm(
                    pt.pow(pt.tensor(x).float(),2),
                    pt.pow(self.v,2)
                )
            ),
            dim=1
        )
        # print("pt.pow(pt.mm(pt.tensor(x).float(),self.v))",pt.pow(pt.mm(pt.tensor(x).float(),self.v)[0:10,1],2))
        # print("pt.mm(pt.pow(pt.tensor(x).float(),2),pt.pow(self.v,2))",pt.mm(pt.pow(pt.tensor(x).float(),2),pt.pow(self.v,2))[0:10,1])
        # print("pair_interactions shape",pair_interactions.shape)
        # print("pair_interactions",pair_interactions[0:10])
        # print("w0 shape",self.w0.shape)
        output1=pt.add(self.w0.transpose(1,0),linear_term.transpose(1,0))
        # print("output1 shape",output1.shape)
        output2=pt.ceil(pt.add(output1,0.5*pair_interactions)/3)
        # output=self.w0+linear_term+0.5*pair_interactions
        return output2

    def forward(self,x):
        output=self.fm_layer(x)
        return output

k=40   #因子的数目
fm=FM_model(n,p,k)
fm
# print(len(list(fm.parameters())))

#训练网络
optimizer=pt.optim.SGD(fm.parameters(),lr=0.01)   #学习率为0.01
# optimizer.zero_grad()   #如果不置零，Variable的梯度在每次backwrd的时候都会累加
# x_train=pt.autograd.Variable(pt.tensor(x_train),requires_grad=True)
# y_train=pt.autograd.Variable(pt.tensor(y_train),requires_grad=True)
for i in range(20):
    optimizer.zero_grad()  # 如果不置零，Variable的梯度在每次backwrd的时候都会累加
    output=fm(x_train)
    # 处理内存溢出的问题
    # try:
    #     output=fm(x_train)
    # except RuntimeError as exception:
    #     if "not enough memory" in str(exception):
    #         print("WARING:out of memeory")
    #         if hasattr(pt.cuda,'empty_cache'):
    #             pt.cuda.empty_cache()
    #             output = fm(x_train)
    #     else:
    #         raise exception
    # output=pt.autograd.Variable(output,requires_grad=True)
    output=output.transpose(1,0)
    # print("output shape",output.shape)
    #平方差
    loss_func= pt.nn.MSELoss()
    # mse_loss=loss_func(output,pt.tensor((output.shape,1)))
    mse_loss=loss_func(output,pt.tensor(y_train).float())
    l2_regularization=pt.tensor(0).float()
    # print("l2_regularization type",l2_regularization.dtype)
    #打印参数数量
    # print(fm.parameters())
    # print(len(list(fm.parameters())))
    #加入l2正则
    for param in fm.parameters():
        # print("param type",pt.norm(param,2).dtype)
        l2_regularization+=pt.norm(param,2)
    # loss=mse_loss
    # print(loss)
    loss=mse_loss+l2_regularization
    loss.backward()
    optimizer.step()   #进行更新
    # print(loss)

print(loss)
print(y_train[0:10],"  ",output[0:10])
#精度函数

#测试网络


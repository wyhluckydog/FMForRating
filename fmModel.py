import pandas as pd
import torch as pt
from sklearn.feature_extraction import DictVectorizer
import test

class FM_model(pt.nn.Module):
    def __init__(self,n,p,k):
        super(FM_model, self).__init__()
        self.n=n   #测试用例的数目
        self.p=p
        self.k=k
        self.linear=pt.nn.Linear(self.n,1,bias=True)
        self.v=pt.randn((self.p,self.k),requires_grad=True)
        self.w=pt.randn((self.p,1),requires_grad=True)
        # self.v=pt.normal(0,0.01,(self.p,self.k))
        # self.w=pt.normal(0,0.1,(self.p,1))
        self.w0=pt.randn((self.n,1),requires_grad=True)

    def fm_layer(self,x):
        #FM算法的线性部分
        linear_term=pt.mm(pt.tensor(x).float(),self.w)
        print("linear_term shape",linear_term.shape)
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
        print("pair_interactions shape",pair_interactions.shape)
        print("pair_interactions",pair_interactions[0:10])
        # print("pair_interactions",pair_interactions[0:10])
        print(self.w0.shape)
        output = self.w0.transpose(1,0) + linear_term.transpose(1,0) + 0.5 * pair_interactions
        # output=self.w0+linear_term+0.5*pair_interactions.resize_(self.n,1)
        return output

    def forward(self,x):
        output=self.fm_layer(x)
        return output

k=40   #因子的数目
fm=FM_model(test.n,test.p,k)
fm
# print(len(list(fm.parameters())))

#训练网络
optimizer=pt.optim.SGD(fm.parameters(),lr=0.01)   #学习率为0.01
optimizer.zero_grad()   #如果不置零，Variable的梯度在每次backwrd的时候都会累加
# x_train=pt.autograd.Variable(pt.tensor(x_train),requires_grad=True)
# y_train=pt.autograd.Variable(pt.tensor(y_train),requires_grad=True)
output=fm(test.x_train)
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
print("output shape",output.shape)
# print(y_train[0:10],"  ",output[0:10])
#平方差
loss_func= pt.nn.MSELoss()
# mse_loss=loss_func(output,pt.tensor((output.shape,1)))
mse_loss=loss_func(output,pt.tensor(test.y_train).float())
l2_regularization=pt.tensor(0).float()
# print("l2_regularization type",l2_regularization.dtype)
#打印参数数量
# print(fm.parameters())
# print(len(list(fm.parameters())))
#加入l2正则
# for param in fm.parameters():
#     print("param type",pt.norm(param,2).dtype)
#     l2_regularization+=pt.norm(param,2)
loss=mse_loss
# print(loss)
# loss=mse_loss+l2_regularization
loss.backward()
optimizer.step()   #进行更新

print(loss)
#精度函数

#测试网络
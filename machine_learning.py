from user_sklearn import trainData
import user_vectools as vt
from matplotlib import pyplot as plt
from sklearn import preprocessing as proc
import numpy as n
import sys


data = n.load("samples/fm_abs_at_stack_2dortho_1.npy")
print(n.shape(data))
# sys.exit(0)
data = n.abs(data)
target = n.load("temp/target.npy")
data2 = list()
for i in range(len(data)):
	# data[i,:] = data[i,:]/n.max(n.abs(data[i,:]))
	data2.append(vt.put_mat_in_vec(data[i,:].reshape(115,20000)[57-10:114-10,2000:20000-2000]))
# # # plt.pcolor(n.real(data[50,:].reshape(15,10000)))
# # # plt.show()
# # # sys.exit(0)

data = n.array(data2)
data = proc.scale(data,axis=1)
# print(n.shape(data))


train = trainData(data,target)
clf,params,perf = train.try_model("Ridge",Niter=100,with_plot=True,dim_reduction=["PCA",None])
print(perf)
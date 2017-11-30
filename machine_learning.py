from user_sklearn import trainData
import user_vectools as vt
from sklearn import preprocessing as proc
import numpy as n
import sys


data = n.load("vec_fourier.npy")
print(n.shape(data))
# sys.exit(0)
target = n.load("temp/target.npy")
# data2 = list()
# for i in range(len(data)):
# 	# data[i,:] = data[i,:]/n.max(n.abs(data[i,:]))
# 	data2.append(vt.put_mat_in_vec(data[i,:].reshape(115,10001)[57:57+10,:]))

# data = n.array(data2)

# data = proc.scale(data,axis=1)

train = trainData(data,target)
clf,params,perf = train.try_model("SVM",with_plot=True,dim_reduction=["PCA",None])
print(perf)
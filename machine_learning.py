from user_idas import DasData
import numpy as n
from matplotlib import pyplot as plt
from scipy import signal as sig
import time
import sys
from sklearn import preprocessing as proc,svm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Lasso,ElasticNet
from sklearn.metrics import r2_score
from multiprocessing import Pool

filepath = {"data/water/1inch/03082017/":["2","3","4","5","6","7","8","9","10","11","12"],"data/water/1inch/03102017/":["10","13"]}

fourier_mat = list()
sum_int = list()
dif_sum_int = list()
target = list()

for i,j in filepath.items():
	for ii in j:
		data_fname = ii+"m3h.tdms"
		f = DasData(i+data_fname,"temp")
		for iii in range(6):
			target.append(int(ii))
			res,kvec,fvec = f.fft2_calc("1","z2",int(10*iii),1,10,fft_type="rfft",norm=None,crop=[-f.fsamp/2/1300,f.fsamp/2/1300,0,10001],resamp=None,val_type="abs",max_lim=None)
			res= n.real(res)
			res = res/n.max(res)
			fourier_mat.append(n.reshape(res,int(n.shape(res)[0]*n.shape(res)[1])))
			argsdat = [[1300,1700,kvec,fvec,res,1,1],[1300,1700,kvec,fvec,res,1,-1]]
			t = time.time()
			pool = Pool()
			ss = pool.starmap(f.get_sum_with_speed_range,argsdat)
			speed = ss[0][0]
			s1 = ss[0][1]
			s2 = ss[1][1]
			print(time.time()-t)
			dif_sum_int.append(n.abs(s1-s2)/n.max(n.abs(s1-s2)))
			s=n.concatenate((n.array(s1),n.array(s2)))
			s = s/n.max(s)
			sum_int.append(s)


fourier_mat = n.array(fourier_mat)
sum_int = n.array(sum_int)
dif_sum_int = n.array(dif_sum_int)
target = n.array(target)
n.save("temp/fourier_mat",fourier_mat)
n.save("temp/sum_int",sum_int)
n.save("temp/dif_sum_int",dif_sum_int)
n.save("temp/target",target)

# xtrain, xtest,ytrain,ytest = train_test_split(vecs,vecy,test_size=0.33)
fmtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
sixtrain, sixtest,siytrain,siytest = train_test_split(sum_int,target,test_size=0.33)
dsixtrain, dsixtest,dsiytrain,dsiytest = train_test_split(dif_sum_int,target,test_size=0.33)



alpha = n.arange(0,0.002,0.0001)
lasso_res = list()
for i in alpha:
	lasso = Lasso(alpha=i)
	y_pred_lasso = lasso.fit(xtrain, ytrain).predict(xtest)
	lasso_res.append(r2_score(ytest, y_pred_lasso))


lasso_max = n.array(lasso_res).argmax()
print(alpha[lasso_max],lasso_res[lasso_max])
# print("r^2 on test data : %f" % r2_score_lasso)

# #############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.figure()
inc1 = n.abs(y_test-y_pred_lasso)/y_test
plt.plot(y_test,inc1,"x")
fit2 = n.poly1d(n.polyfit(y_test,inc1,8))
plt.plot(n.sort(y_test),fit2(n.sort(y_test)),"-")
plt.xlabel("Desired Flowrate")
plt.ylabel("Relative Uncertainty")
plt.grid()
plt.title("LASSO Score = "+str(r2_score_lasso))

plt.figure()
inc2 = n.abs(y_test-y_pred_enet)/y_test
plt.plot(y_test,inc2,"x")
fit2 = n.poly1d(n.polyfit(y_test,inc2,8))
plt.plot(n.sort(y_test),fit2(n.sort(y_test)),"-")
plt.xlabel("Desired Flowrate")
plt.ylabel("Relative Uncertainty")
plt.grid()
plt.title("Elastic NET Score = "+str(r2_score_enet))
	# res = proc.MinMaxScaler().fit_transform(n.real(res))

	# res=f.set_to_lim(res,min_lim=[0.7,0])

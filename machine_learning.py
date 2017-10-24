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


l = ["2","3","4","5","6","7","8","9","10","11","12"]
data_dir = "data/water/1inch/03082017/"
vec = list()
vecs = list()
vecss = list()
vecy= list()
for ii in l:
	t = time.time()
	data_fname = ii+"m3h.tdms"
	f = DasData(data_dir+data_fname,"temp")
	tt = time.time()	
	print("data loaded in " + str(tt-t) + "s")
	for iii in range(6):
		vecy.append(int(ii))
		res,kvec,fvec = f.fft2_calc("1","z2",int(10*iii),1,10,fft_type="rfft",norm="ortho",crop=[-f.fsamp/2/1300,f.fsamp/2/1300,0,10001],resamp=None,val_type="abs",max_lim=None)
		res = res/n.max(res)
		vec.append(n.reshape(res,int(n.shape(res)[0]*n.shape(res)[1])))
		speed,s1 = f.get_sum_with_speed_range(1300,1700,kvec,fvec,n.real(res),signe=1)
		speed,s2 = f.get_sum_with_speed_range(1300,1700,kvec,fvec,n.real(res),signe=-1)
		s1 = s1/n.max(s1)
		s2 = s2/n.max(s2)
		vecss.append(n.abs(s1-s2))
		s=n.concatenate((n.array(s1),n.array(s2)))
		vecs.append(s)


l = ["10","13"]
data_dir = "data/water/1inch/03102017/"
for ii in l:
	t = time.time()
	data_fname = ii+"m3h.tdms"
	f = DasData(data_dir+data_fname,"temp")
	tt = time.time()
	print("data loaded in " + str(tt-t) + "s")
	for iii in range(6):
		vecy.append(int(ii))
		res,kvec,fvec = f.fft2_calc("1","z2",int(10*iii),1,10,fft_type="rfft",norm="ortho",crop=[-f.fsamp/2/1300,f.fsamp/2/1300,0,10001],resamp=None,val_type="abs",max_lim=None)
		res = res/n.max(res)
		vec.append(n.reshape(res,int(n.shape(res)[0]*n.shape(res)[1])))
		speed,s1 = f.get_sum_with_speed_range(1300,1700,kvec,fvec,n.real(res),signe=1)
		speed,s2 = f.get_sum_with_speed_range(1300,1700,kvec,fvec,n.real(res),signe=-1)
		s1 = s1/n.max(s1)
		s2 = s2/n.max(s2)
		vecss.append(n.abs(s1-s2))
		s=n.concatenate((n.array(s1),n.array(s2)))
		vecs.append(s)


vec = n.array(vec)
vecs = n.array(vecs)
vecss = n.array(vecs)
vecy = n.array(vecy)

# xtrain, xtest,ytrain,ytest = train_test_split(vecs,vecy,test_size=0.33)
xtrain, xtest,ytrain,ytest = train_test_split(vec,vecy,test_size=0.33)
# vecy = n.array([7,8,9,10,11,12])
vec2 = list()
l = ["10","13"]
data_dir = "data/water/1inch/03102017/"
for ii in l:
	t = time.time()
	data_fname = ii+"m3h.tdms"
	f = DasData(data_dir+data_fname,"temp")
	tt = time.time()
	print("data loaded in " + str(tt-t) + "s")
	for iii in range(6):
		res,kvec,fvec = f.fft2_calc("1","z2",int(iii*10),1,10,fft_type="rfft",norm="ortho",crop=[-f.fsamp/2/1300,f.fsamp/2/1300,0,10001],resamp=None,val_type="abs",max_lim=None)
		res = res/n.max(res)
		# vec.append(n.reshape(res,int(n.shape(res)[0]*n.shape(res)[1])))
		vec2.append(n.reshape(res,int(n.shape(res)[0]*n.shape(res)[1])))

xtrain, xtest, ytrain, ytest = train_test_split(vecss, vecy, test_size=0.33)

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

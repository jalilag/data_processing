import numpy as n
from matplotlib import pyplot as plt
from scipy import signal as sig
import time
import sys
from sklearn import preprocessing as proc,svm
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV 
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from user_sklearn import user_grid_search
from multiprocessing import cpu_count

# if __name__== "__main__":
fourier_mat = n.load("temp/fourier_mat.npy")
# fourier_mat = n.load("temp/sum_int.npy")
# dif_sum_int = n.load("temp/dif_sum_int.npy")
target = n.load("temp/target.npy")
fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
# sixtrain, sixtest,siytrain,siytest = train_test_split(sum_int,target,test_size=0.33)
# dsixtrain, dsixtest,dsiytrain,dsiytest = train_test_split(dif_sum_int,target,test_size=0.33)

model = "RF"
dim_reduction = "None"
err = 0.01

# PCA
if dim_reduction == "PCA":
	pca = PCA(n_components=78)
	dim = pca.fit(fourier_mat)
	fourier_mat = dim.transform(fourier_mat)

# Lasso
if model == "Lasso":
	## Find params
	params = {"alpha":{"start":0,"end":1}}
	clf = user_grid_search(Lasso(),fourier_mat,target,params,err)
	print("Best params for Lasso")
	for ii,jj in clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()].items():
		print(ii,": {0:9.9f}".format(jj))
	## Test model
	params = clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()]
	ares = list()		
	for i in range(100):	
		fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
		clf = Lasso(alpha=params["alpha"])
		clf_pred = clf.fit(fmxtrain, fmytrain)	
		lres = list()
		for j in range(100):
			fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
			# clf = Lasso(alpha=n.mean(ares))		
			lres.append(r2_score(fmytest, clf_pred.predict(fmxtest)))
		ares.append(n.mean(lres))
		print(ares[-1])
	print("Mean res",n.mean(ares))
	print(params)

# ElasticNet
if model == "ElasticNet":
	## Find params
	params = {"alpha":{"start":0.00001,"end":1},"l1_ratio":{"start":0,"end":1}}
	clf = user_grid_search(ElasticNet(),fourier_mat,target,params,err)
	print("Best params for Lasso")
	for ii,jj in clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()].items():
		print(ii,": {0:9.9f}".format(jj))
	## Test model
	params = clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()]
	ares = list()		
	for i in range(100):	
		fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
		clf = ElasticNet(alpha=params["alpha"],l1_ratio=params["l1_ratio"])
		clf_pred = clf.fit(fmxtrain, fmytrain)	
		lres = list()
		for j in range(100):
			fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
			lres.append(r2_score(fmytest, clf_pred.predict(fmxtest)))
		ares.append(n.mean(lres))
		print(ares[-1])
	print("Mean res",n.mean(ares))
	print(params)


# Ridge
if model == "Ridge":
	## Find params
	params = {"alpha":{"start":0,"end":1}}
	clf = user_grid_search(Ridge(),fourier_mat,target,params,err)
	print("Best params for Lasso")
	for ii,jj in clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()].items():
		print(ii,": {0:9.9f}".format(jj))
	## Test model
	params = clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()]
	ares = list()		
	for i in range(100):	
		fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
		clf = Ridge(alpha=params["alpha"])
		clf_pred = clf.fit(fmxtrain, fmytrain)	
		lres = list()
		for j in range(100):
			fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
			# clf = Lasso(alpha=n.mean(ares))		
			lres.append(r2_score(fmytest, clf_pred.predict(fmxtest)))
		ares.append(n.mean(lres))
		print(ares[-1])
	print("Mean res",n.mean(ares))
	print(params)


# SVR
if model == "SVR":
	## Find params
	params = {"epsilon":{"start":0,"end":1},"C":{"start":0.1,"end":1000}}
	clf = user_grid_search(svm.SVR(kernel="linear"),fourier_mat,target,params,err)
	print("Best params for Lasso")
	for ii,jj in clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()].items():
		print(ii,":",jj)
	## Test model
	params = clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()]
	ares = list()		
	for i in range(100):	
		fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
		clf = svm.SVR(kernel="linear",epsilon=params["epsilon"],C=params["C"])
		clf_pred = clf.fit(fmxtrain, fmytrain)	
		lres = list()
		for j in range(100):
			fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
			# clf = Lasso(alpha=n.mean(ares))		
			lres.append(r2_score(fmytest, clf_pred.predict(fmxtest)))
		ares.append(n.mean(lres))
		print(ares[-1])
	print("Mean res",n.mean(ares))
	print(params)


# Random Forest
if model == "RF":
	## Find params
	params = {"n_estimators":n.arange(2,100,1)}
	clf = user_grid_search(RandomForestRegressor(),fourier_mat,target,params,err)
	print("Best params for RF")
	for ii,jj in clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()].items():
		print(ii,": {0:9.9f}".format(jj))
	## Test model
	params = clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()]
	ares = list()		
	for i in range(100):	
		fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
		clf = RandomForestRegressor(n_estimators=params["n_estimators"])
		clf_pred = clf.fit(fmxtrain, fmytrain)	
		lres = list()
		for j in range(100):
			fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(fourier_mat,target,test_size=0.33)
			# clf = Lasso(alpha=n.mean(ares))		
			lres.append(r2_score(fmytest, clf_pred.predict(fmxtest)))
		ares.append(n.mean(lres))
		print(ares[-1])
	print("Mean res",n.mean(ares))
	print(params)

## Figures
plt.figure()
plt.subplot(211)
plt.plot(target,n.abs(clf_pred.predict(fourier_mat)-target)/target*100,"x")
plt.xlabel("Flowrates (m3/h)")
plt.ylabel("%")
plt.title("Relative uncertainty")
plt.subplot(212)
plt.plot(target,clf_pred.predict(fourier_mat),"x")
plt.xlabel("Desired Flowrates (m3/h)")
plt.ylabel("Measured Flowrates (m3/h)")
plt.show()


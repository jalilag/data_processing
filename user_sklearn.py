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
from multiprocessing import cpu_count

class trainData():
	fourier_mat = None
	target = None
	err = None	

	def __init__(self,data_file,target_file,err=0.01):
		self.fourier_mat = n.real(data_file)
		self.target = n.real(target_file)
		self.err = err

	def try_model(self,model="Lasso",model_params=None,dim_reduction=None,with_plot=False,Niter=100):
		# PCA
		if dim_reduction is not None:
			if dim_reduction[0] == "PCA":
				pca = PCA(n_components=dim_reduction[1])
				dim = pca.fit(self.fourier_mat)
				self.fourier_mat = dim.transform(self.fourier_mat)


		# Models
		## Find params
		if model_params is None:
			if model == "Lasso":
				params = {"alpha":{"start":0,"end":1000}}
				clf = self.user_grid_search(Lasso(),params,self.err)
			if model == "ElasticNet":
				params = {"alpha":{"start":0.00001,"end":1000},"l1_ratio":{"start":0,"end":1}}
				clf = self.user_grid_search(ElasticNet(),params,self.err)
			if model == "Ridge":
				params = {"alpha":{"start":0,"end":1000}}
				clf = self.user_grid_search(Ridge(),params,self.err)
			if model == "SVR" or model == "SVM":
				params = {"epsilon":{"start":0,"end":1},"C":{"start":0.1,"end":1000}}
				clf = self.user_grid_search(svm.SVR(kernel="linear"),params,self.err)
			if model == "RF":
				params = {"n_estimators":n.arange(2,100,1)}
				clf = self.user_grid_search(RandomForestRegressor(),params,self.err)
			
			print("Best params for",model)
			for ii,jj in clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()].items():
				print(ii,": {0:9.9f}".format(jj))
			params = clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()]
		else:
			params = model_params
		## Test model
		ares = list()		
		for i in range(Niter):	
			fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(self.fourier_mat,self.target,test_size=0.33)
			if model == "Lasso": clf = Lasso(**params)
			if model == "ElasticNet": clf = ElasticNet(**params)
			if model == "Ridge": clf = Ridge(**params)
			if model == "SVR" or model == "SVM": clf = svm.SVR(kernel="linear",**params)
			if model == "RF": clf = RandomForestRegressor(**params)
			clf_pred = clf.fit(fmxtrain, fmytrain)	
			lres = list()
			for j in range(Niter):
				fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(self.fourier_mat,self.target,test_size=0.33)
				lres.append(r2_score(fmytest, clf_pred.predict(fmxtest)))
			ares.append(n.mean(lres))
			print(ares[-1])

		s = 1-n.sum(n.abs(clf_pred.predict(self.fourier_mat)-self.target)/self.target)/len(self.target)
		print(s)
		## Figures
		if with_plot:
			plt.figure()
			plt.subplot(211)
			plt.plot(self.target,n.abs(clf_pred.predict(self.fourier_mat)-self.target)/self.target*100,"x")
			plt.xlabel("Flowrates (m3/h)")
			plt.ylabel("Relative uncertainty%")
			plt.title(model + " model" + " -> "+"{:.2%}".format(s))

			plt.grid()
			plt.subplot(212)
			# plt.title("Model : "+model)
			plt.plot(self.target,clf_pred.predict(self.fourier_mat),"x")
			plt.xlabel("Desired Flowrates (m3/h)")
			plt.ylabel("Predicted Flowrates (m3/h)")
			plt.grid()
			plt.show()
		if dim_reduction is not None:
			return clf_pred,dim,params,[n.mean(ares),n.sqrt(n.var(ares))]
		else:
			return clf_pred,params,[n.mean(ares),n.sqrt(n.var(ares))]

	def user_grid_search(self,skmod,params,max_err=0.01,n_jobs=1):
		parameters = dict()
		err = 10
		for i,j in params.items():
			if isinstance(j,dict):
				parameters[i] = n.arange(j["start"],j["end"]+n.abs(j["end"]-j["start"])/10,n.abs(j["end"]-j["start"])/10)
			else:
				if isinstance(j,list) or isinstance(j,n.ndarray):
					parameters[i] = j
				else:
					parameters[i] = [j] 
		print(parameters)
		while err > max_err :
			clf = GridSearchCV(skmod,parameters,n_jobs=n_jobs)
			clf.fit(self.fourier_mat,self.target)
			N1 = clf.cv_results_["mean_test_score"].argmax()
			err = n.sqrt(n.var(clf.cv_results_["mean_test_score"]))
			print(clf.cv_results_["mean_test_score"])
			print("Max mean score : ",n.max(clf.cv_results_["mean_test_score"]))
			print("Err : ",err)
			for i,j in clf.cv_results_["params"][N1].items():
				if isinstance(params[i],list) or isinstance(params[i],n.ndarray): 
					err = 0
					break
				if not isinstance(j,str):
					if len(parameters[i]) > 1:
						dx = n.abs(parameters[i][1]-parameters[i][0])
						if j-dx < parameters[i][0]: par1 = j
						else: par1 = j-dx
						if j+dx > parameters[i][-1]: par2 = j
						else: par2 = j+dx
						parameters[i]= n.arange(par1,par2,n.abs(par1-par2)/10)
					else:

						parameters[i] = [j]
				else:
					parameters[i] = [j]
				print(i,parameters[i])
				print(i+" max :",j)
		return clf

	def user_score(self,target,pred):
		S = 0
		N = len(target)
		for i in range(N):
			S += n.abs(target[i]-pred[i])/target[i]
		return S/N
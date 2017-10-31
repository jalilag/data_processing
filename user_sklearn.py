import numpy as n
from matplotlib import pyplot as plt
from scipy import signal as sig
import time
import sys
from sklearn import preprocessing as proc,svm
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def user_grid_search(skmod,data,target,params,max_err=0.01,n_jobs=1):
	parameters = dict()
	err = 10
	for i,j in params.items():
		if isinstance(j,dict):
			parameters[i] = n.arange(j["start"],j["end"],n.abs(j["end"]-j["start"])/10)
		else:
			if isinstance(j,list) or isinstance(j,n.ndarray):
				parameters[i] = j
			else:
				parameters[i] = [j] 
	print(parameters)
	while err > max_err :
		clf = GridSearchCV(skmod,parameters,n_jobs=n_jobs)
		clf.fit(data,target)
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


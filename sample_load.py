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

if __name__=="__main__":
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
				t = time.time()
				res,kvec,fvec = f.fft2_calc("1","z2",int(10*iii),1,10,fft_type="rfft",norm=None,crop=[-f.fsamp/2/1300,f.fsamp/2/1300,0,10001],resamp=None,val_type="abs",max_lim=None)
				print("Fourier transform executed in " + str(time.time()-t) + " s")
				res= n.real(res)
				res = res/n.max(res)
				fourier_mat.append(n.reshape(res,int(n.shape(res)[0]*n.shape(res)[1])))
				argsdat = [[1300,1700,kvec,fvec,res,1,1],[1300,1700,kvec,fvec,res,1,-1]]
				t = time.time()
				pool = Pool(2)
				ss = pool.starmap(f.get_sum_with_speed_range,argsdat)
				pool.close()
				pool.join()
				speed = ss[0][0]
				s1 = ss[0][1]
				s2 = ss[1][1]
				del ss
				del pool
				del argsdat
				print("Radial scan executed in " + str(time.time()-t) + " s")
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





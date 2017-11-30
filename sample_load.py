from user_idas import DasData
import numpy as n
from matplotlib import pyplot as plt
from scipy import signal as sig
import scipy as sc
from scipy.interpolate import interp2d
import time
import sys
from sklearn import preprocessing as proc
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Lasso,ElasticNet
from sklearn.metrics import r2_score
from multiprocessing import Pool
import cv2
import user_vectools as vt
# import skimage

if __name__=="__main__":
	filepath = {"data/water/1inch/03082017/":["2","3","4","5","6","7","8","9","10","11","12"],
				"data/water/1inch/03102017/":["10","13"]}
	# filepath = {"data/water/1inch/03082017/":["9"]}
	fourier_mat = list()
	# fourier_mat2 = list()
	save_dir = "samples/"
	save_file="turbulence"
	version ="9"
	target = list()
	for i,j in filepath.items():
		for ii in j:
			data_fname = ii+"m3h.tdms"
			f = DasData(i+data_fname,"temp")
			for iii in range(0,60,10):
				target.append(int(ii))
				t = time.time()
				pp_stack = None #{1:{"func":vt.normL3,"args":None}}
				pp_fftt =  None # {1:{"func":vt.normL3,"args":None}}
				pp_ffts =  None #{1:{"func":vt.normL3,"args":None}}
				res,kvec,fvec = f.fft2_calc("1","z2",int(iii),1,10,fft_type="rfft",norm="ortho",crop=None,val_type="real",pp_fftt=pp_fftt,pp_ffts=pp_ffts,pp_stack=pp_stack)
				fourier_mat.append(vt.put_mat_in_vec(res))
				# res,kvec,fvec = f.quarter_sum(res,kvec,fvec)
				# res,[kvec,fvec] = vt.crop_vecs(res,[kvec,fvec],[0,20,0,40])
				# fourier_mat2.append(f.put_mat_in_vec(res))
	# f.plot_fft(res,kvec,fvec)
	# res2 = f.partial_quarter_sum(res,kvec,fvec,upper_right=[-1,1,0,0])
	# f.plot_fft(res2,kvec,fvec)
	# res3,[k3,f3] = vt.crop_vecs(res2,[kvec,fvec],[0,None,0,None])
	# f.plot_fft(res3,k3,f3)

	# res4 = vt.normL3(proc.scale(res3))
	# f.plot_fft(res4,k3,f3)
	# res2 = f.partial_quarter_sum(res,kvec,fvec,upper_right=[1,-1,0,0])
	# f.plot_fft(res2,kvec,fvec)
	# res2 = f.partial_quarter_sum(res,kvec,fvec,upper_right=[0,1,-1,0])
	# f.plot_fft(res2,kvec,fvec)
	# res2 = f.partial_quarter_sum(res,kvec,fvec,upper_right=[0,1,0,-1])
	# f.plot_fft(res2,kvec,fvec)
	# f.get_sum_with_speed_range(1,10,k3,f3,res3,0.5)
	# plt.show()
	# sys.exit(0)
	n.save(save_dir+save_file+"_"+version,n.array(fourier_mat))
	# n.save(save_dir+save_file+"_resized_"+version,n.array(fourier_mat2))
	# n.save(save_dir+save_file+"_ppfftt_"+version,pp_fftt)
	# n.save(save_dir+save_file+"_ppffts_"+version,pp_ffts)
	# vt.save_obs("samples")
		# n.save("temp/sum_int2",sum_int)
		# n.save("temp/dif_sum_int",dif_sum_int)
		# n.save("temp/target",target)





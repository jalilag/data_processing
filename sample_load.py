from user_idas import DasData, read_param_data,mean_over_time_range
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
	# filepath = {"data/water/1inch/03082017/":["2","3","4","5","6","7","8","9","10","11","12"],
	# 			"data/water/1inch/03102017/":["10","13"]}
	filepath = {"data/water/1inch/06122017/":["max"]}
	# filepath = {"data/water/1inch/03082017/":["10"]}
	fourier_mat = list()
	save_dir = "samples/"
	save_file="new_mat_abs_at_stack_2dortho"
	version ="1"
	startat = 0
	duration = 60
	stack = 10
	target = list()
	for i,j in filepath.items():
		for ii in j:
			data_fname = ii
			if ii.isdigit(): data_fname += "m3h"
			mat_param,ent,start,end = read_param_data(i+data_fname+".txt")
			print(start,end)
			f = DasData(i+data_fname+".tdms","temp")
			for iii in range(startat,duration,stack):
				target.append(mean_over_time_range(mat_param,iii,startat+duration))
				t = time.time()
				pp_stack = {1:{"func":n.abs,"args":None}}
				pp_fftt = None #{1:{"func":vt.highlight_freq,"args":{"fvec":n.sort(n.fft.fftfreq(20000,1/20000)),"fhigh":500,"flow":-500,"coef":0.001}}}
				pp_ffts =  None #{1:{"func":proc.scale,"args":None}}
				res,kvec,fvec = f.fft2_calc("1","z2",int(iii),1,10,fft_type="fft",norm="ortho",crop=None,val_type=None,pp_fftt=pp_fftt,pp_ffts=pp_ffts,pp_stack=pp_stack)
				fourier_mat.append(vt.put_mat_in_vec(res))
				#[-f.fsamp/2/1300,f.fsamp/2/1300,None,None]
				# res,kvec,fvec = f.quarter_sum(res,kvec,fvec)
				# res,[kvec,fvec] = vt.crop_vecs(res,[kvec,fvec],[0,20,0,40])
				# fourier_mat2.append(f.put_mat_in_vec(res))
	print(target)
	sys.exit(0)
	# plt.figure()
	# plt.pcolor(res)
	# plt.figure()
	# plt.pcolor(proc.scale(res))
	# plt.show()
	f.plot_fft(res,kvec,fvec)
	# f.plot_fft(res,fvec,kvec)
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
	plt.show()
	sys.exit(0)
	n.save(save_dir+save_file+"_"+version,n.array(fourier_mat))
	# n.save(save_dir+save_file+"_resized_"+version,n.array(fourier_mat2))
	# n.save(save_dir+save_file+"_ppfftt_"+version,pp_fftt)
	# n.save(save_dir+save_file+"_ppffts_"+version,pp_ffts)
	# vt.save_obs("samples")
		# n.save("temp/sum_int2",sum_int)
		# n.save("temp/dif_sum_int",dif_sum_int)
		# n.save("temp/target",target)





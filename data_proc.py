from user_idas import DasData
import numpy as n
from matplotlib import pyplot as plt
from scipy import signal as sig
import time
import sys
from sklearn import preprocessing as proc


# l = ["2","3","4","5","6"]
l = ["10"]
for ii in l:
	t = time.time()
	data_dir = "data/water/1inch/03082017/"
	data_fname = ii+"m3h.tdms"
	f = DasData(data_dir+data_fname,"temp")
	f.plot_raw_data(10,15)
	tt = time.time()
	print("data loaded in " + str(tt-t) + "s")
	res,kvec,fvec = f.fft2_calc("1","z2",0,1,60,fft_type="rfft",norm="ortho",crop=[-f.fsamp/2/1300,f.fsamp/2/1300,0,10001],resamp=None,val_type="abs",max_lim=None)
	# res = proc.MinMaxScaler().fit_transform(n.real(res))
	# res=f.set_to_lim(res,min_lim=[0.7,0])
	res = res/n.max(res)
	print("max",n.max(res))
	print("FFT mean in ",str(time.time()-tt),"s")
	plt.figure()
	t =time.time()
	speed,s = f.get_sum_with_speed_range(1300,1700,kvec,fvec,n.real(res),signe=1)
	# s = f.get_max_slope(kvec,fvec,res,1)
	# print("max speed",speed[n.abs(s-n.max(s)).argmin()])
	print("speed in " + str(time.time()-t))
	plt.plot(speed,s,"x")
	step = 10
	# vec = f.mobil_cumulativ_vec(speed,s,step)
	# plt.plot(vec[0:,0],vec[0:,1]/step)
	# for i in range(2,9):
	fit1 = n.poly1d(n.polyfit(speed,s,8))
	vmax1 = speed[fit1(speed).argmax()] 
	vm1 = speed[s.argmax()]
	print("vmax = ",vmax1,vm1) 
	plt.plot(speed,fit1(speed),"-")
	plt.plot(vmax1,fit1(vmax1),"o")
	plt.grid()
	t =time.time()
	speed,s = f.get_sum_with_speed_range(1300,1700,kvec,fvec,n.real(res),signe=-1)
	# vec = f.mobil_cumulativ_vec(speed,s,step)
	# plt.plot(vec[0:,0],vec[0:,1]/step)
	vm2 = speed[s.argmax()]

	# s = f.get_max_slope(kvec,fvec,res,-1)
	# print("max speed",speed[n.abs(s-n.max(s)).argmin()])
	print("speed in " + str(time.time()-t))
	plt.plot(speed,s,"x")

	# for i in range(2,9):
	# fit2 = n.polyfit(speed,s,8)
	fit2 = n.poly1d(n.polyfit(speed,s,8))
	# print(fit2)
	plt.plot(speed,fit2(speed),"-")
	vmax2 = speed[fit2(speed).argmax()]
	print("vmax = ",vmax2,vm2) 
	plt.plot(vmax2,fit2(vmax2),"o")
	# plt.legend(["Up side","fit Up side","Down side","fit Down side"])
	# plt.legend
	# v = 
	plt.title("Velocity poly = "+str(n.abs(vmax1 - vmax2)/2)+" m/s " +"\n" +"Velocity data = "+str(n.abs(vm1-vm2)/2)+" m/s " +"\n" )
	plt.savefig(data_dir+ii+"_bis_intensity_speed.png")
	plt.figure()
	res,kvec,fvec = f.resamp_vecs(res,kvec,fvec,100,100)
	x,y = n.meshgrid(fvec,kvec)
	plt.pcolor(x,y,n.real(res))
	plt.colorbar()
	plt.savefig(data_dir+ii+"_bis_matrice.png")
	plt.close()

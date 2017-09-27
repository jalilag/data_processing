from user_idas import DasData
import numpy as n
from matplotlib import pyplot as plt
from scipy import signal as sig
import time

def clearall(except_var=None):
	all = [var for var in globals() if var[0] != "_"]
	for var in all:
		if except_var is not None:
			if var not in except_var:
				del globals()[var]

if __name__ == '__main__':
	t = time.time()
	# f = DasData("data/water_1in/optical/10_170803154549.tdms","temp")
	f = DasData("data/water_10m3h_1inch_170803154549.tdms")
	tt = time.time()
	print("data loaded in " + str(tt-t) + "s")
	k = [10]
	for i in k:
		# res,kvec,fvec = f.fft2_quick_calc(230,330,0,0.1,i,fft_type="fft",resamp=None,norm=None,crop=[-10,10,-1,8000],min_lim=0)
		# res,kvec,fvec = f.fft2_quick_calc(230,330,0,0.1,i,fft_type="fft",resamp=None,norm=None,crop=[-10,10,-8000,8000],min_lim=0)
		res,kvec,fvec = f.fft2_calc(230,330,0,0.1,i,fft_type="fft",resamp=None,norm=None,crop=[-10,10,-8000,8000],min_lim=0)
		print("FFT mean in ",str(time.time()-tt),"s")

		res,kvec,fvec = f.resamp_vecs(res,kvec,fvec,[100,100])
	# fvec = sig.resample(fvec,100)
	# fig,ax1 = plt.subplots(1,1)
		x,y = n.meshgrid(fvec,kvec)
		plt.pcolor(x,y,n.real(res))
		plt.colorbar()
		plt.savefig("real_"+str(i)+".png")
		plt.clf()
		plt.close()
		plt.pcolor(x,y,n.abs(res))
		plt.colorbar()
		plt.savefig("abs_"+str(i)+".png")
		plt.clf()
		plt.close()
		# clearall()


# ax2.pcolor(n.abs(res))
# N= 10
# ax1.set_yticks(n.arange(N))
# ax1.set_xticks(n.arange(N))
# ax1.set_yticklabels(sig.resample(kvec,N).astype(int))
# ax1.set_xticklabels(sig.resample(fvec,N).astype(int))
# ax.set_yticklabels(kvec.astype(int))
# ax.set_xticklabels(fvec.astype(int))
# ax1.set_xticklabels(kvec)
# ax1.set_yticklabels(fvec)
# plt.imshow(abs(res))

# res = 
# # Lecture des propriétés
# params = tdms_file.object()
# # for name, value in rooto.properties.items():
# #     print("{0}: {1}".format(name, value))
# # Liste des groupes
# # print("\n","Liste des groupes :","\n",tdms_file.groups(),"\n")
# # Liste des channels d'un groupe
# # print("Liste des channels d'un groupe","\n",tdms_file.group_channels('Measurement'),"\n")
# print(tdms_file.object())

# channels = tdms_file.group_channels('Measurement')


# Liste des données d'un channel
# channel = tdms_file.object('Untitled','Untitled')
# data = channel.data
# time = channel.time_track()
# print("DATA","\n",data,"\n")
# print("TIME","\n",time,"\n")
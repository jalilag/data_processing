from user_tdms import UTdms
import numpy as n
from matplotlib import pyplot as plt
# import pyfftw
import time
from scipy import signal,ndimage
from multiprocessing import Pool,Process,cpu_count
import sys
import os

class DasData(UTdms):
	filename =""
	fsamp = 0
	spatial_res = 0
	duration = 0
	start_pos = 0
	measure_length = 0
	end_pos = 0
	fibre_index = 0
	fibre_corr = 0
	Nl = 0
	Nt = 0

	lines = {
		"1":{"z1":[188,262],"z2":[325,440],"z3":[530,640],"dist_ratio":118.15879},
		"1.5":{"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":170.169602},
		"2":{"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":212.057504},
		"2.5":{"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":256.388867},
		"3":{"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":311.89037},
		"liner":{"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":311.89037},
		"pmma":{"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":311.89037},
	}
	def __init__(self,file, memmap_dir=None):
		t = time.time()	
		super().__init__(file, memmap_dir=None)
		self.filename = self.get_param("name")

		self.fsamp = int(self.get_param("SamplingFrequency[Hz]"))
		self.spatial_res = float(self.get_param("SpatialResolution[m]"))
		self.duration = float(self.get_param("StreamTime[s]"))
		if self.duration == float("Inf"):
			self.duration = float(len(self.get_channel(0,0).data)/self.fsamp)
		self.start_pos = float(self.get_param("StartPosition[m]"))
		self.measure_length = float(self.get_param("MeasureLength[m]"))
		self.end_pos = self.start_pos + self.measure_length
		self.fibre_corr = float(self.get_param("Fibre Length Multiplier"))
		self.fibre_index = float(self.get_param("FibreIndex"))
		self.Nl = int(self.spatial_res * self.measure_length)
		self.Nt = int(self.fsamp*self.duration)
		print("Data loaded in " + str(time.time()-t) + " s")	

	def get_array(self,xfrom,xend,yfrom,yend):
		"""Extract an array from a TDMS file"""
		if xend < xfrom or yend < yfrom or xend > self.Nl or yend > self.Nt:
			print("Erreur indice de matrice")
			return list()
		Nx = xend-xfrom
		Ny = yend-yfrom
		res = n.zeros((Nx,Ny))
		for i in range(xfrom,xend):
			res[i-xfrom,0:Ny] = self.get_channel(0,i).data[yfrom:yend] 
		return res

	# def fft2_calc(self,pos_start,pos_end,start_time,mean_time,coef_time=1,fft_type="fft",norm=None,dist_corr=116,resamp=None,crop=None,val_sum=None,min_lim=None,max_lim=None):
	def fft2_calc(self,line,zone,start_time,mean_time,coef_time=1,fft_type="fft",norm=None,resamp=None,crop=None,val_sum=None,min_lim=None,max_lim=None,val_type=None):
		""" Calculation of 2d fft
			pos_start,pos_end: define the position to consider
			start_time:        Time where the fft begin
			mean_time:         Duration of the signal on which is calculated   
			coef_time:         How many sets are used
			fft_type:          fft,rfft,ifft,irfft
			norm:			   None or ortho
			dist_corr:		   Relation between fiber length and pipe length
			resamp:            [lsampling,timesampling]
			crop:              [kinf,ksup,finf,fsup]
			val_sum:           Intensity are summed or averaged (default averaged)
			min_lim:           Set to 0 intensity less than min_lim
		"""
		c = 0
		start_data = int(start_time*self.fsamp)
		dt = int(mean_time*self.fsamp)
		duration = int(mean_time*coef_time*self.fsamp)
		if fft_type == "fft" or fft_type == "ifft":
			res = n.zeros((int(self.lines[line][zone][1]-self.lines[line][zone][0]),dt),dtype=complex)
		if fft_type == "rfft" or fft_type == "irfft":
			res = n.zeros((int(self.lines[line][zone][1]-self.lines[line][zone][0]),int(dt/2)+1),dtype=complex)

		print("start time; i; duration")
		for i in range(start_data,int(start_data+duration),dt):
			print(start_data,i,duration)
			if fft_type == "fft": res += self.set_to_lim(n.fft.fft2(self.get_array(self.lines[line][zone][0],self.lines[line][zone][1],i,int(i+dt)),norm=norm),min_lim,max_lim)
			if fft_type == "rfft": res += self.set_to_lim(n.fft.rfft2(self.get_array(self.lines[line][zone][0],self.lines[line][zone][1],i,int(i+dt)),norm=norm),min_lim,max_lim)
			if fft_type == "ifft": res += self.set_to_lim(n.fft.ifft2(self.get_array(self.lines[line][zone][0],self.lines[line][zone][1],i,int(i+dt)),norm=norm),min_lim,max_lim)
			if fft_type == "irfft": res += self.set_to_lim(n.fft.irfft2(self.get_array(self.lines[line][zone][0],self.lines[line][zone][1],i,int(i+dt)),norm=norm),min_lim,max_lim)
			c+= 1
		print(n.shape(res))
		if val_type == "real": res = n.real(res)
		if val_type == "abs": res = n.abs(n.real(res))
		dx = self.spatial_res/self.lines[line]["dist_ratio"]
		if val_sum is not None: res = res/c
		kvec = n.fft.fftfreq(n.size(res,0),dx)
		if fft_type == "fft" or fft_type == "ifft": 
			fvec = n.fft.fftfreq(n.size(res,1),1/self.fsamp)
		if fft_type == "rfft" or fft_type == "irfft":
			fvec = n.fft.rfftfreq((n.size(res,1)-1)*2+1,1/self.fsamp)
		if crop is not None:
			res,kvec,fvec = self.crop_vecs(res,kvec,fvec,crop)
		print(n.shape(res))
		if resamp is not None:
			if resamp[0] == "auto": resamp[0] = len(fvec)
			if resamp[1] == "auto": resamp[1] = len(kvec)
			res,kvec,fvec = self.resamp_vecs(res,kvec,fvec,resamp[0],resamp[1],False)
		res,kvec,fvec = self.sort_vecs(res,kvec,fvec)
		return res,kvec,fvec

	def fft2_calc_par(self,pos_start,pos_end,start_time,mean_time,coef_time=1,fft_type="fft",norm=None,dist_corr=116,resamp=None,crop=None,val_sum=None,min_lim=None):
		""" Calculation of 2d fft
			pos_start,pos_end: define the position to consider
			start_time:        Time where the fft begin
			mean_time:         Duration of the signal on which is calculated   
			coef_time:         How many sets are used
			fft_type:          fft,rfft,ifft,irfft
			norm:			   None or ortho
			dist_corr:		   Relation between fiber length and pipe length
			resamp:            [lsampling,timesampling]
			crop:              [kinf,ksup,finf,fsup]
			val_sum:           Intensity are summed or averaged (default averaged)
			min_lim:           Set to 0 intensity less than min_lim
		"""
		start_data = int(start_time*self.fsamp)
		dt = int(mean_time*self.fsamp)
		duration = int(mean_time*coef_time*self.fsamp)
		if fft_type == "fft" or fft_type == "ifft":
			res = n.zeros((int(pos_end-pos_start),dt),dtype=complex)
		if fft_type == "rfft" or fft_type == "irfft":
			res = n.zeros((int(pos_end-pos_start),int(dt/2)+1),dtype=complex)
		print("start time; i; duration")
		respool = list()
		k = coef_time
		for i in range(start_data,int(start_data+duration),k*dt):
			pool = Pool()
			for l in range(k):
				if int(i+(l+1)*dt) <= int(start_data+duration):
					argsdat = list()
					argsdat.append((self.get_array(pos_start,pos_end,i+l*dt,int(i+(l+1)*dt)),norm))
			# if len(argsdat) > 0:
				# pool = Pool()
				# res += sum(pool.starmap(n.fft.rfft2,argsdat,100000000))
				# p=Process(target=n.fft.rfft2,args)

		dx = self.spatial_res/dist_corr
		if val_sum is None: res = res/coef_time
		kvec = n.fft.fftfreq(n.size(res,0),dx)
		if fft_type == "fft" or fft_type == "ifft": 
			fvec = n.fft.fftfreq(n.size(res,1),1/self.fsamp)
		if fft_type == "rfft" or fft_type == "irfft":
			fvec = n.fft.rfftfreq((n.size(res,1)-1)*2+1,1/self.fsamp)

		if crop is not None:
			res,kvec,fvec = self.crop_vecs(res,kvec,fvec,crop)
		if min_lim is not None:
			for i in range(n.size(res,0)):
				for j in range(n.size(res,1)):
					if res[i,j] < min_lim: res[i,j] = 0
		if resamp is not None:
			res,kvec,fvec = self.resamp_vecs(res,kvec,fvec,100,100,False)
		res,kvec,fvec = self.sort_vecs(res,kvec,fvec)
		return res,kvec,fvec

	def fft2_quick_calc(self,pos_start,pos_end,start_time,mean_time,coef_time=1,fft_type="fft",norm=None,dist_corr=116,resamp=None,crop=None,val_sum=None,min_lim=None):
		""" Calculation of 2d fft
			pos_start,pos_end: define the position to consider
			start_time:        Time where the fft begin
			mean_time:         Duration of the signal on which is calculated   
			coef_time:         How many sets are used
			fft_type:          fft,rfft,ifft,irfft
			norm:			   None or ortho
			dist_corr:		   Relation between fiber length and pipe length
			resamp:            [lsampling,timesampling]
			crop:              [kinf,ksup,finf,fsup]
			val_sum:           Intensity are summed or averaged (default averaged)
			min_lim:           Set to 0 intensity less than min_lim
		"""
		nthread = cpu_count()
		c = 0
		start_data = int(start_time*self.fsamp)
		dt = int(mean_time*self.fsamp)
		duration = int(mean_time*coef_time*self.fsamp)
		if fft_type == "fft" or fft_type == "ifft":
			res = n.zeros((int(pos_end-pos_start),dt),dtype=complex)#).astype("complex")
		if fft_type == "rfft" or fft_type == "irfft":
			res = n.zeros((int(pos_end-pos_start),int(dt/2)+1),dtype=float)
		print("start time; i; duration")
		for i in range(start_data,int(start_data+duration),dt):
			print(start_data,i,duration)
			# if fft_type == "fft": res += n.fft.fft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=norm)
			# if fft_type == "rfft": res += n.fft.rfft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=norm)
			# if fft_type == "ifft": res += n.fft.ifft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=norm)
			# if fft_type == "irfft": res += n.fft.irfft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=norm)
			# res += pyfftw.FFTW(self.get_array(pos_start,pos_end,i,int(i+dt)),axis=(0,1))
			# pyfftw.forget_wisdom()
			# dat = 
			# b4= n.zeros_like(res)
			# pyfftw.FFTW( self.get_array(pos_start,pos_end,i,int(i+dt)).astype("complex"), b4, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE', ), threads=nthread, planning_timelimit=None )()
			res += pyfftw.builders.fft2(self.get_array(pos_start,pos_end,i,int(i+dt)).astype("complex"), s=None, axes=(-2, -1), overwrite_input=False, planner_effort='FFTW_ESTIMATE', threads=nthread, auto_align_input=False, auto_contiguous=False, avoid_copy=True)()
			# res += pyfftw.interfaces.numpy_fft.fft2(self.get_array(pos_start,pos_end,i,int(i+dt)).astype("complex"), s=None, axes=(-2, -1), norm=None, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=8, auto_align_input=True, auto_contiguous=True)()
			# fft()
			# res += fft()
			# res += b4
			# del fft;del b4
			c+= 1
		dx = self.spatial_res/dist_corr
		if val_sum is None: res = res/c
		kvec = n.fft.fftfreq(n.size(res,0),dx)
		if fft_type == "fft" or fft_type == "ifft": 
			fvec = n.fft.fftfreq(n.size(res,1),1/self.fsamp)
		if fft_type == "rfft" or fft_type == "irfft":
			fvec = n.fft.rfftfreq((n.size(res,1)-1)*2+1,1/self.fsamp)
		if crop is not None:
			res,kvec,fvec = self.crop_vecs(res,kvec,fvec,crop)
		if min_lim is not None:
			data = self.set_to_lim(data,min_lim)
		if resamp is not None:
			res,kvec,fvec = self.resamp_vecs(res,kvec,fvec,100,100,False)
		res,kvec,fvec = self.sort_vecs(res,kvec,fvec)
		return res,kvec,fvec


	def resamp_vecs(self,data,xvec=None,yvec=None,N1=100,N2=100,sort_data=True):
		if N1 is not None:
			# print(data,N1)
			data = signal.resample(data,N1,axis=0)
			print("done")
		if N2 is not None:
			data = signal.resample(data,N2,axis=1)
			print("done")
		if xvec is not None and N1 is not None:
			xvec = signal.resample(xvec,N1)
			print("done")
		if yvec is not None and N2 is not None:
			yvec = signal.resample(yvec,N2)
			print("done")
		if sort_data and xvec is not None and yvec is not None:
			data,xvec,yvec = self.sort_vecs(data,xvec,yvec)
			print("done")
		return data,xvec,yvec

	def resamp_raw_data(self,data,N):
		return ndimage.zoom(data,N/len(data))


	def sort_vecs(self,data,xvec,yvec):
		xsort = xvec.argsort()
		xvec = xvec[xsort]
		data = data[xsort]
		ysort = yvec.argsort()
		yvec = yvec[ysort]
		for i in range(n.size(xvec)):
			data[i] = data[i][ysort]
		return data,xvec,yvec

	def crop_vecs(self,data,xvec,yvec,crop):
		xsort = n.where((xvec > crop[0]) & (xvec < crop[1]))[0]
		ysort = n.where((yvec > crop[2]) & (yvec < crop[3]))[0]
		xvec = xvec[xsort]
		yvec = yvec[ysort]
		data = data[xsort]
		data2 = n.zeros((n.size(xvec),n.size(yvec)),dtype=complex)
		for i in range(n.size(data,0)):
			data2[i] = data[i][ysort]
		return data2,xvec,yvec

	def set_to_lim(self,data,min_lim = None,max_lim=None):
		if min_lim is not None or max_lim is not None:
			N = n.shape(data)
			for i in range(N[0]):
				if min_lim is not None:
					data[i][n.where(data[i]<min_lim[0])]= min_lim[1]
				if max_lim is not None:
					data[i][n.where(data[i]>max_lim[0])]= max_lim[1]
		return data

	def get_kid_by_freq_speed(self,kvec,fvec,speed,signe=1):
		idx = n.abs(kvec-0).argmin()
		if signe == 1: kvec = kvec[idx:]
		else: kvec = kvec[0:idx+1]
		l = n.zeros((0),dtype=int)
		t = time.time()

		for i in fvec:
			if signe == -1:
				l=n.append(l,n.abs(kvec-signe*float(i/speed)).argmin())
			else:
				l=n.append(l,n.abs(kvec-signe*float(i/speed)).argmin()+idx)
		# print("Kid speed in "+str(time.time()-t)+"s")
		return l

	def get_freq_by_kvec_speed(self,kvec,fvec,speed,signe=1):
		idx = n.abs(kvec-0).argmin()
		if signe == 1: kvec = kvec[idx:]
		else: kvec = kvec[0:idx]
		l = list()
		# ll = n.zeros((0),dtype=int)
		t = time.time()

		for i in range(len(kvec)):
			# if signe == -1:
			val = n.abs(fvec-n.abs(float(i*speed))).argmin()
			if val != 9999:
				l.append([i,val])
			# print(i,speed,i*speed,l[-1])
			# else:
				# l=n.append(l,n.abs(vec-signe*float(i/speed)).argmin()+idx)
		# print("Kid speed in "+str(time.time()-t)+"s")
		# print(speed)
		# print(l)
		# print(kvec,fvec[l],l)

		return n.array(l)


	def sum_from_vecid(self,vecid,res):
		return n.sum(res[vecid,n.arange(res.shape[1])])

	def sum_from_fvecid(self,vecid,res,idx=0):
		print(vecid)
		return n.sum(res[vecid[0:,0]+idx,vecid[0:,1]])


	def get_sum_with_speed_range(self,vmin,vmax,kvec,fvec,res,vstep = 1,signe=1):
		"""
			Calcul la somme associée à une vitesse à f cst
		"""
		c = -1
		speed = n.arange(vmin,vmax,vstep)
		s = n.zeros((len(speed)))
		for i in speed:
			if i%100 == 0: print("speed = "+str(i))
			c+=1
			s[c] = self.sum_from_vecid(self.get_kid_by_freq_speed(kvec,fvec,i,signe),res)
		return speed,s

	def get_sum_with_speed_range2(self,vmin,vmax,kvec,fvec,res,vstep = 1,signe=1):
		"""
			Calcul la somme associée à une vitesse a k cst
		"""
		if signe == 1: idx = n.abs(kvec-0).argmin()
		else: idx =0
		c = -1
		speed = n.arange(vmin,vmax,vstep)
		s = n.zeros((len(speed)))
		for i in speed:
			if i%100 == 0: print("speed = "+str(i))
			c+=1
			s[c] = self.sum_from_fvecid(self.get_freq_by_kvec_speed(kvec,fvec,i,signe),res,idx)
		return speed,s


	def plot_raw_data(self,t_start,t_end,x_by=1,t_by=10):
		print(int(self.end_pos),int(t_start*self.fsamp),int(t_end*self.fsamp))
		plt.pcolor(self.get_array(25,self.Nl,int(t_start*self.fsamp),int(t_end*self.fsamp))[::x_by,::t_by])
		plt.colorbar()
		plt.show()

	def get_max_slope(self,kvec,fvec,res,signe=1):
		l = list()
		idx = n.abs(kvec-0).argmin()
		if signe == 1: kvec = kvec[idx:]
		else: kvec = kvec[0:idx+1]
		for i in range(len(fvec)):
			if signe == 1: val =kvec[res[idx:,i].argmax()-idx]
			else: val = kvec[res[0:idx+1,i].argmax()]
			if val != 0: l.append(i/val)
			else: l.append(0)

		return l


	def mobil_cumulativ_vec(self,vecx,vecy,step):
		l=list()
		for i in range(0,len(vecx)-step,step):
			l.append([n.abs(vecx[i+step]+vecx[i])/2,n.sum(vecy[i:i+step])])
		return n.array(l) 
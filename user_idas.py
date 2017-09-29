from user_tdms import UTdms
import numpy as n
# import pyfftw 
from scipy import signal
from multiprocessing import Pool,Process,cpu_count
import sys

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

	def __init__(self,file, memmap_dir=None):
		super().__init__(file, memmap_dir=None)
		self.filename = self.get_param("name")
		self.fsamp = int(self.get_param("SamplingFrequency[Hz]"))
		self.spatial_res = float(self.get_param("SpatialResolution[m]"))
		self.duration = float(self.get_param("StreamTime[s]"))
		self.start_pos = float(self.get_param("StartPosition[m]"))
		self.measure_length = float(self.get_param("MeasureLength[m]"))
		self.end_pos = self.start_pos + self.measure_length
		self.fibre_corr = float(self.get_param("Fibre Length Multiplier"))
		self.fibre_index = float(self.get_param("FibreIndex"))
		self.Nl = int(self.spatial_res * self.measure_length)
		self.Nt = int(self.fsamp*self.duration)

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

	def fft2_calc(self,pos_start,pos_end,start_time,mean_time,coef_time=1,fft_type="fft",norm=None,dist_corr=116,resamp=None,crop=None,val_sum=None,min_lim=None,max_lim=None):
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
			res = n.zeros((int(pos_end-pos_start),dt),dtype=complex)
		if fft_type == "rfft" or fft_type == "irfft":
			res = n.zeros((int(pos_end-pos_start),int(dt/2)+1),dtype=complex)
		print(n.shape(res))
		print("start time; i; duration")
		for i in range(start_data,int(start_data+duration),dt):
			print(start_data,i,duration)
			if fft_type == "fft": res += n.fft.fft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=norm)
			if fft_type == "rfft": res += n.fft.rfft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=norm)
			if fft_type == "ifft": res += n.fft.ifft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=norm)
			if fft_type == "irfft": res += n.fft.irfft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=norm)
			c+= 1
			res = self.set_to_lim(res,min_lim,max_lim)
		dx = self.spatial_res/dist_corr
		if val_sum is None: res = res/c
		kvec = n.fft.fftfreq(n.size(res,0),dx)
		if fft_type == "fft" or fft_type == "ifft": 
			fvec = n.fft.fftfreq(n.size(res,1),1/self.fsamp)
		if fft_type == "rfft" or fft_type == "irfft":
			fvec = n.fft.rfftfreq((n.size(res,1)-1)*2+1,1/self.fsamp)
		if crop is not None:
			res,kvec,fvec = self.crop_vecs(res,kvec,fvec,crop)
		if resamp is not None:
			res,kvec,fvec = self.resamp_vecs(res,kvec,fvec,resamp[0],resamp[1],False)
		res,kvec,fvec = self.sort_vecs(res,kvec,fvec)
		print(kvec,n.size(kvec))
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


	def resamp_vecs(self,data,xvec,yvec,N1=100,N2=100,sort_data=True):
		data = signal.resample(data,N1,axis=0)
		data = signal.resample(data,N2,axis=1)
		# pool = Pool()
		xvec = signal.resample(xvec,N1)
		yvec = signal.resample(yvec,N2)
		# xvec,yvec = pool.starmap(signal.resample,[(xvec,N1),(yvec,N2)])
		if sort_data: 
			data,xvec,yvec = self.sort_vecs(data,xvec,yvec)
		return data,xvec,yvec

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

	def set_to_lim(self,data,minlim = None,maxlim=None):
		if minlim is not None and maxlim is not None:
			N = n.shape(data)
			for i in range(N[0]):
				if len(N) > 1:
					for j in range(N[1]):
						if minlim is not None:
							if data[i,j] < min_lim[0]: data[i,j] = minlim[1]
						if maxlim is not None:
							if data[i,j] > max_lim[0]: data[i,j] = maxlim[1]
				else:
					if minlim is not None:
						if data[i] < min_lim[0]: data[i] = minlim[1]
					if maxlim is not None:
						if data[i] > max_lim[0]: data[i] = maxlim[1]
		return data

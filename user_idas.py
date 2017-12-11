from user_tdms import UTdms
import numpy as n
from matplotlib import pyplot as plt
from matplotlib import cm
import pyfftw as fftw
# import cv2
import time
import datetime
from scipy import signal
from scipy import ndimage as nd
from multiprocessing import Pool,Process,cpu_count
from sklearn import preprocessing as proc
import sys
import os
# import pywt
import user_vectools as vt

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
		"1":    {"z1":[188,262],"z2":[325,440],"z3":[530,640],"dist_ratio":118.15879},
		"1.5":  {"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":170.169602},
		"2":    {"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":212.057504},
		"2.5":  {"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":256.388867},
		"3":    {"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":311.89037},
		"liner":{"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":311.89037},
		"pmma": {"z1":[180,280],"z2":[241,361],"z3":[446,640],"dist_ratio":311.89037},
	}
	def __init__(self,file, memmap_dir=None):
		t = time.time()	
		super().__init__(file, memmap_dir=None)
		self.filename = self.get_param("name")
		print("Working file :",self.filename)

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
		print(n.shape(res))
		return res

	def fft2_calc(self,line,zone,start_time,mean_time,coef_time=1,fft_type="fft",norm=None,resamp=None,crop=None,val_sum=None,min_lim=None,max_lim=None,val_type=None,data_filter=None,pp_stack=None,pp_fftt=None,pp_ffts=None):
		""" Calculation of 2d fft
			pos_start,pos_end: define the position to consider
			start_time:        Time where the fft begin
			mean_time:         Duration of the signal on which FFT is calculated   
			coef_time:         How many sets are stacked
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
		if fft_type == "fft" or fft_type == "ifft" or fft_type == "fft2":
			res = n.zeros((int(self.lines[line][zone][1]-self.lines[line][zone][0]),dt),dtype=complex)
		if fft_type == "rfft" or fft_type == "irfft" or fft_type == "rfft2":
			res = n.zeros((int(self.lines[line][zone][1]-self.lines[line][zone][0]),int(dt/2)+1),dtype=complex)
			# if fft_type == "rfft2":
			# 	res = n.zeros((int((self.lines[line][zone][1]-self.lines[line][zone][0])/2)+1,dt),dtype=complex)
		k = cpu_count()
		print("FFT time")
		for i in range(start_data,int(start_data+duration),k*dt):
			argsdat = list()
			print(i)
			for l in range(k):
				if int(i+(l+1)*dt) <= int(start_data+duration):
					if fft_type == "rfft2" or fft_type == "fft2":
						argsdat.append((self.get_array(self.lines[line][zone][0],self.lines[line][zone][1],i,int(i+dt)),None,(-2,-1),norm))
					else:
						argsdat.append((self.get_array(self.lines[line][zone][0],self.lines[line][zone][1],i,int(i+dt)),None,1,norm))
			pool = Pool(k)
			t=time.time()
			if fft_type == "fft": restemp = pool.starmap(n.fft.fft,argsdat)
			if fft_type == "rfft": restemp = pool.starmap(n.fft.rfft,argsdat)
			if fft_type == "ifft": restemp = pool.starmap(n.fft.ifft,argsdat)
			if fft_type == "irfft": restemp = pool.starmap(n.fft.irfft,argsdat)
			if fft_type == "fft2": restemp = pool.starmap(n.fft.fft2,argsdat)
			if fft_type == "rfft2": restemp = pool.starmap(n.fft.rfft2,argsdat)
			pool.close()
			pool.join()
			print("FFT time finished in ",time.time()-t,"s")
			del pool
			del argsdat
			c+= len(restemp)
			restemp = n.array(restemp)
			for j in range(len(restemp)):
				aa = vt.set_to_lim(restemp[j,:,:],min_lim,max_lim)
				if pp_stack is not None:
					for i,j in pp_stack.items():
						if j["args"] is not None: aa = j["func"](n.real(aa),**j["args"])
						else: aa = j["func"](n.real(aa))
				res += aa 
			del restemp
		# X and Y vec calc
		if fft_type == "fft" or fft_type == "ifft" or fft_type == "fft2": 
			fvec = n.fft.fftfreq(n.size(res,1),1/self.fsamp)
		if fft_type == "rfft" or fft_type == "irfft" or fft_type == "rfft2":
			fvec = n.fft.rfftfreq((n.size(res,1)-1)*2+1,1/self.fsamp)
		if fft_type != "rfft2" or fft_type != "fft2":
			res,[kvec,fvec] = vt.sort_vecs(res,[None,fvec])
		dx = self.spatial_res/self.lines[line]["dist_ratio"]
		kvec = n.fft.fftfreq(n.size(res,0),dx)
		# Post fft time processing
		if pp_fftt is not None:
			for i,j in pp_fftt.items():
				if j["args"] is not None: res = j["func"](n.real(res),**j["args"])
				else: res = j["func"](n.real(res))
		print("FFT time calc in progress")
		t=time.time()
		if fft_type != "fft2" and fft_type != "rfft2":
			res = n.fft.fft(res,None,0,norm)
		print("FFT space calc finished in",time.time()-t,"s")
		# Post fft space processing
		if pp_ffts is not None:
			for i,j in pp_ffts.items():
				if j["args"] is not None: res = j["func"](n.real(res),**j["args"])
				else: res = j["func"](n.real(res))

		# Post processing
		if val_type == "real": res = n.real(res)
		if val_type == "abs": res = n.abs(n.real(res))
		if val_sum is not None: res = res/c
		if crop is not None:
			res,[kvec,fvec] = vt.crop_vecs(res,[kvec,fvec],crop)
		if resamp is not None:
			if resamp[0] == "auto": resamp[0] = len(fvec)
			if resamp[1] == "auto": resamp[1] = len(kvec)
			res,[kvec,fvec] = vt.resamp_vecs(res,[kvec,fvec],[resamp[0],resamp[1]],False)
		res,[kvec,fvec] = vt.sort_vecs(res,[kvec,fvec])
		return res,kvec,fvec



	def resamp_raw_data(self,data,N):
		return ndimage.zoom(data,N/len(data))


	def get_kid_by_freq_speed(self,kvec,fvec,knorm,speed,signe=1):
		idx = n.abs(kvec-0).argmin()
		if signe == 1: kvec = kvec[idx:]
		else: kvec = kvec[0:idx+1]
		l = n.zeros((0),dtype=int)

		for i in fvec[n.where(fvec>0)]:
			kx = float(i/speed)
			if kx<=n.max(kvec) and n.sqrt(kx*kx+i*i) < knorm: 
				fid = n.abs(kvec-signe*kx).argmin()
				if fid != idx: 
					if signe == -1 :
						l=n.append(l,fid)
					else:
						if fid != idx: l=n.append(l,fid+idx)
				else:
					l= n.append(l,None)
		return l

	def get_kf_with_speed(self,kvec,fvec,knorm,speed,signe=1,tol=0.03):
		l = n.zeros((0),dtype=int)
		fres = list()
		kres = list()
		kid = list()
		for i in n.where(fvec>0)[0]:
			kx = float(fvec[i]/speed)
			if kx<=n.max(kvec) and n.sqrt(kx*kx+fvec[i]*fvec[i]) <= knorm: 
				fid,err = self.get_val_in_vec(kvec,kx)
				if err < tol and fid != -1:
					fres.append(i)
					kres.append(fid)
					kid.append(kx)
		return kres,fres,kid

	def get_val_in_vec(self,vec,val):
		err = 1
		fid=-1
		for i in range(len(vec)):
			err2 = n.abs(vec[i]-val)/val
			if err2 < err and n.sign(vec[i]) == n.sign(val):
				err = err2
				fid = i
		if n.sign(vec[fid]) == -1:
			sys.exit(0)
		return fid,err

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


	def sum_from_vecid(self,vecid,fvec,res):
		idx = n.abs(fvec-0).argmin()
		s = 0
		for i in range(len(vecid)):
			if vecid[i] is not None:
				# print(vecid[i],fvec[i+idx+1],res[vecid[i],i+idx+1])
				s+=res[vecid[i],i]
		return s

	def sum_from_fvecid(self,vecid,res,idx=0):
		print(vecid)
		return n.sum(res[vecid[0:,0]+idx,vecid[0:,1]])


	def get_sum_with_speed_range(self,vmin,vmax,kvec,fvec,res,vstep = 1,signe=1):
		"""
			Calcul la somme associée à une vitesse à f cst
		"""
		c = -1
		speed = n.arange(vmin,vmax,vstep)
		knorm = n.max(kvec)
		s = n.zeros((len(speed)))
		plt.figure()
		cc = list()
		kk = list()
		ff = list()
		for i in speed:
			c+=1
			kres,fres,kid = self.get_kf_with_speed(kvec,fvec,knorm,i,signe)

			cc.append(len(kres))

			plt.plot(fvec[fres],kid,"-k")
			kk.append(kres)
			ff.append(fres)
		N = -1
		print(N)
		for i in range(len(speed)):
			s[i] = n.sum(res[kk[i][0:N],ff[i][0:N]])
			plt.plot(fvec[ff[i][0:N]],kvec[kk[i][0:N]],"x-")
			print(kvec[kk[i][0:N]])
		plt.grid()
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

	def plot_fft(self,data,kvec,fvec,Nsx=None,Nsy=None,title=None):
		plt.figure()
		data,[kvec,fvec] = vt.resamp_vecs(data,[kvec,fvec],[Nsx,Nsy])
		x,y = n.meshgrid(fvec,kvec)
		plt.pcolor(x,y,n.real(data))
		plt.colorbar()
		if title is not None: plt.title(title)
		# plt.show()

	def butter_filter(self,data,fcut,Norder=5):
		fcut = fcut/self.fsamp/2		
		b,a = signal.butter(Norder,fcut,btype="lowpass")
		zi = signal.lfilter_zi(b,a)
		return signal.lfilter(b,a,data,zi=zi)
		# return signal.filtfilt(b,a,data)


	def set_bandwidth(self,data,xvec,boundaries,axis=-1,val2set=0):
		if axis == -1:
			ysort = n.where((xvec > boundaries[0]) & (xvec < boundaries[1]))[0]
			for i in range(n.size(data,0)):
				data[i][ysort] = val2set
		elif axis == -2:
			xsort = n.where((xvec > boundaries[0]) & (xvec < boundaries[1]))[0]
			data[xsort] = val2set
		return data

	def quarter_sum(self,res,kvec,fvec,valtype="real"):		
		k0 = n.abs(kvec-0).argmin()
		f0 = n.abs(fvec-0).argmin()
		Nk = len(kvec[k0:])
		Nf = len(fvec[f0:])
		Nx = n.shape(res)[0]
		Ny = n.shape(res)[1]
		res2 = n.zeros((Nk,Nf))
		for i in range(Nk):
			for j in range(Nf):
				if valtype == "real":
					res2[i,j] = n.real(res[k0+i,f0+j])+n.real(res[k0-i,f0+j]) + n.real(res[k0+i,f0-j]) + n.real(res[k0-i,f0-j])
				elif valtype == "abs":
					res2[i,j] = n.abs(res[k0+i,f0+j])+n.abs(res[k0-i,f0+j]) + n.abs(res[k0+i,f0-j]) + n.abs(res[k0-i,f0-j])
		return res2,kvec[k0:],fvec[f0:]	

	def partial_quarter_sum(self,res,kvec,fvec,upper_left=[0,0,0,0],upper_right=[0,0,0,0],down_left=[0,0,0,0],down_right=[0,0,0,0]):		
		k0 = n.abs(kvec-0).argmin()
		f0 = n.abs(fvec-0).argmin()
		Nk = len(kvec[k0:])
		Nf = len(fvec[f0:])
		Nx = n.shape(res)[0]
		Ny = n.shape(res)[1]
		res2 = n.zeros((Nx,Ny))
		for i in range(Nk):
			for j in range(Nf):
				res2[k0+i,f0-j] = upper_left[0]*res[k0+i,f0-j]+upper_left[1]*res[k0+i,f0+j]+upper_left[2]*res[k0-i,f0-j]+upper_left[3]*res[k0-i,f0+j]
				res2[k0+i,f0+j] = upper_right[0]*res[k0+i,f0-j]+upper_right[1]*res[k0+i,f0+j]+upper_right[2]*res[k0-i,f0-j]+upper_right[3]*res[k0-i,f0+j]
				res2[k0-i,f0-j] = down_left[0]*res[k0+i,f0-j]+down_left[1]*res[k0+i,f0+j]+down_left[2]*res[k0-i,f0-j]+down_left[3]*res[k0-i,f0+j]
				res2[k0-i,f0+j] = down_right[0]*res[k0+i,f0-j]+down_right[1]*res[k0+i,f0+j]+down_right[2]*res[k0-i,f0-j]+down_right[3]*res[k0-i,f0+j]
		return res2	

	def delete_quarter(self,res,kvec,fvec,coef=[1,0,0,1]):		
		k0 = n.abs(kvec-0).argmin()
		f0 = n.abs(fvec-0).argmin()
		Nx = n.shape(res)[0]
		Ny = n.shape(res)[1]
		if coef[0] == 1:
			for i in range(k0+1,Nx):
				for j in range(f0):
					res[i,j] = 0
		if coef[1] == 1:
			for i in range(k0+1,Nx):
				for j in range(f0+1,Ny):
					res[i,j] = 0
		if coef[2] == 1:
			for i in range(0,k0):
				for j in range(0,f0):
					res[i,j] = 0
		if coef[3] == 1:
			for i in range(0,k0):
				for j in range(f0+1,Ny):
					res[i,j] = 0
		return res




	def denoise(self,img, weight=0.1, eps=1e-3, num_iter_max=200):
	    """Perform total-variation denoising on a grayscale image.
	    
	    Parameters
	    ----------
	    img : array
	        2-D input data to be de-noised.
	    weight : float, optional
	        Denoising weight. The greater `weight`, the more de-noising (at
	        the expense of fidelity to `img`).
	    eps : float, optional
	        Relative difference of the value of the cost function that determines
	        the stop criterion. The algorithm stops when:
	            (E_(n-1) - E_n) < eps * E_0
	    num_iter_max : int, optional
	        Maximal number of iterations used for the optimization.

	    Returns
	    -------
	    out : array
	        De-noised array of floats.
	    
	    Notes
	    -----
	    Rudin, Osher and Fatemi algorithm.
	    """
	    u = n.zeros_like(img)
	    px = n.zeros_like(img)
	    py = n.zeros_like(img)
	    
	    nm = n.prod(img.shape[:2])
	    tau = 0.125
	    
	    i = 0
	    while i < num_iter_max:
	        u_old = u
	        
	        # x and y components of u's gradient
	        ux = n.roll(u, -1, axis=1) - u
	        uy = n.roll(u, -1, axis=0) - u
	        
	        # update the dual variable
	        px_new = px + (tau / weight) * ux
	        py_new = py + (tau / weight) * uy
	        norm_new = n.maximum(1, n.sqrt(px_new **2 + py_new ** 2))
	        px = px_new / norm_new
	        py = py_new / norm_new

	        # calculate divergence
	        rx = n.roll(px, 1, axis=1)
	        ry = n.roll(py, 1, axis=0)
	        div_p = (px - rx) + (py - ry)
	        
	        # update image
	        u = img + weight * div_p
	        
	        # calculate error
	        error = n.linalg.norm(u - u_old) / n.sqrt(nm)
	        
	        if i == 0:
	            err_init = error
	            err_prev = error
	        else:
	            # break if error small enough
	            if n.abs(err_prev - error) < eps * err_init:
	                break
	            else:
	                e_prev = error
	                
	        # don't forget to update iterator
	        i += 1

	    return u

def read_param_data(fname):
	mat_param = list()
	print(fname)
	try:
		fp = open(fname,"r")
		ent = fp.readline().split("\n")[0].split(";")
		l = fp.read().split("\n")
		fp.close
	except:
		return None
	for line in l:
		if line != "": mat_param.append(line.split(";"))
	mat_param = n.array(mat_param).astype(n.float)
	start = mat_param[0,0]
	end = mat_param[-1,0]	
	mat_param[:,0] = mat_param[:,0]-mat_param[0,0]
	return mat_param,ent,datetime.datetime.fromtimestamp(start),datetime.datetime.fromtimestamp(end)

def mean_over_time_range(mat,start,end,keyid=1):
	return n.mean(mat[n.where((mat[:,0]>=start) & (mat[:,0]<end)),keyid])

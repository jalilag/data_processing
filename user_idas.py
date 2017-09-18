from user_tdms import UTdms
import numpy as n

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
		if xend < xfrom or yend < yfrom or xend > self.Nl or yend > self.Nt:
			print("Erreur indice de matrice")
			return list()
		Nx = xend-xfrom
		Ny = yend-yfrom
		res = n.zeros((Nx,Ny))
		for i in range(xfrom,xend):
			res[i-xfrom,0:Ny] = self.get_channel(0,i).data[yfrom:yend] 
		return res

	def fft_calc(self,pos_start,pos_end,mean_time,end_time,norm=None):
		c = 0
		dt = int(self.fsamp * mean_time)
		res = n.zeros((int(pos_end-pos_start),int(dt/2)+1),dtype=complex)
		for i in range(0,int(end_time*self.fsamp),dt):
			print(dt,i,end_time)
			res += n.fft.rfft2(self.get_array(pos_start,pos_end,i,int(i+dt)),norm=None)
			c+= 1
		K_wave_num = n.fft.fftfreq(n.size(res,0),1)#k=1/lambda
		sort = K_wave_num.argsort()
		K_wave_num = K_wave_num[sort]
		res = res[sort]#trier comme K_wave_num
		return res/c


#working only for iDAS data

#class tdms_data
import nptdms
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import integrate

#class mass_posttraitement
import glob

class tdms_data(nptdms.TdmsFile):
	##TODO: vérifier que la taille du vecteur  charger n'est pas trop grande
	
	def __init__(self, fileNameIn):
		super().__init__(fileNameIn)
	#############################
	##data management functions##
	#############################
	def read_data_info(self):
		# Lecture des propriétés
		rooto = self.object()
		for name, value in rooto.properties.items():
    			print("{0}: {1}".format(name, value))
		# Liste des groupes
		print("\n","Liste des groupes :","\n",self.groups(),"\n")
		#group_name = self.groups
	
	def cpy_tdms_to_file(self, fileNameOut):
		rooto = self.object()
		file_out = open(fileNameOut,"w")
		
		for name, value in rooto.properties.items():
			file_out.write("{0}: {1}".format(name, value))
			file_out.write("\n")
		# Liste des groupes
		file_out.write("\nListe des groupes :\n")
		#TODO file_out.write(str(self.groups()))
		file_out.write("\n")
		#TODO: warning si fichier trop gros
		
		for i in range (0,len(self.group_channels('Measurement'))):
			for j in range(0, len(self.object('Measurement', repr(i)).data)):
				file_out.write(repr(self.object('Measurement', repr(i)).data[j])+" ")
			file_out.write("\n")
	#entré rien ou type list ou type np.ndarray. loc in array index, time in array index.
	def s_limits_to_array(self, channel_start=0, channel_end=None):
		N = len(self.group_channels('Measurement'))
		if channel_end == None:
			channel_end = N
		channel_start = int(channel_start)
		channel_end = int(channel_end)
		if channel_end == channel_start:
			channel_end+=1
		if channel_end < channel_start:
			c = channel_end
			channel_end = channel_start
			channel_start = c
		channels = np.linspace(channel_start, channel_end-1, channel_end-channel_start)
		return channels
	def t_limits_to_array(self, time_start=0, time_end=None):
		T = self.object().property("StreamTime[s]")
		f = self.object().property("SamplingFrequency[Hz]")
		if time_end == None:
			time_end = T*f
		time_start = int(time_start)
		time_end = int(time_end)
		if time_end == time_start:
			time_end+=1
		if time_end < time_start:
			t = time_end
			time_end = time_start
			time_start = t
		times = np.linspace(time_start, time_end-1, time_end-time_start)
		return times
	def limits_to_arrays(self, channel_start=0, channel_end=None, time_start=0, time_end=None):
		times = self.t_limits_to_array(time_start, time_end)
		channels = self.s_limits_to_array(channel_start, channel_end)
		return channels, times

	def load_data(self, loc=None, time=None):
		#TODO verifier que la taille n'est pas excessive
		
		if type(time)==list or type(time) == np.ndarray:
			c = len(time)
			if type(loc) == list or type(loc) == np.ndarray:
				data = np.zeros([len(loc), c])
				j = 0
				for i in loc:
					k=0
					for l in time:
						data[j][k] = self.object('Measurement', repr(int(i))).data[int(l)]
						k+=1
					j+=1
			else:#load all			
				l = len(self.group_channels('Measurement'))
				data = np.zeros([l,c])
				for i in range (0,l):
					k=0
					for j in time:
						data[i][k] = self.object('Measurement', repr(i)).data[int(j)]
						k+=1
		else:#load all
			c = self.object().property("StreamTime[s]")*self.object().property("SamplingFrequency[Hz]")
			if type(loc) == list or type(loc) == np.ndarray:
				data = np.zeros([len(loc), c])
				j = 0		
				for i in loc:
					data[j] = self.object('Measurement', repr(int(i))).data
					j+=1
			else:#load all			
				l = len(self.group_channels('Measurement'))
				data = np.zeros([l,c])
				for i in range (0,l):
					data[i] = self.object('Measurement', repr(i)).data
		return data
	############################
	##data operation functions##
	############################
	#int number, loc=int, 
	def stack(self, data, number):
		return sum(data[0:number])
	#application d'un filtre de Butterworth passe bas
	def filter_lowpass(self, data, data_is_fourier=False, wp=0.0001, ws=0.01, pass_loss_dB=0.01, stop_loss_dB=40, option='butterworth'):
		if not data_is_fourier:
			data_fft = np.fft.rfft(data)
		else:
			data_fft=data
		if option == 'butterworth':
			N, Wn = signal.buttord(wp, ws, pass_loss_dB, stop_loss_dB, False)
			b, a = signal.butter(N, Wn, btype='lowpass', analog=False, output='ba')
			w, h = signal.freqz(b, a, np.size(data_fft,1), whole=False)
		#check filter
		#print(N," = N \n", Wn," = Wn\n", b," = b\n", a," = a\n", w," = w\n", h," = h\n")
		#plt.semilogx(w*f/2/np.pi, np.log10(abs(h)))
		#plt.title('lowpass Filter used')
		#plt.show()			
		#calc Fourier(data)*Fourier(filter)
		for i in range(np.size(data_fft,0)):
			for j in range(np.size(data_fft,1)):
				data_fft[i][j] = data_fft[i][j]*h[j]
		#ifft
		if not data_is_fourier:
			return np.fft.irfft(data_fft)
		else:
			return data_fft
	#bandpass
	def filter_bandpass(self, data, data_is_fourier=False, wp=[0.2,0.5], ws=[0.1,0.6], pass_loss_dB=0.01, stop_loss_dB=40, option='butterworth'):
		if not data_is_fourier:
			data_fft = np.fft.rfft(data)
		else:
			data_fft=data		
		if option == 'butterworth':
			N, Wn = signal.buttord(wp, ws, pass_loss_dB, stop_loss_dB, analog=False)
			b, a = signal.butter(N, Wn, btype='bandpass', analog=False, output='ba')
			w, h = signal.freqz(b, a, np.size(data_fft,1), whole=False)
		#check filter
		#print(N," = N \n", Wn," = Wn\n", b," = b\n", a," = a\n", w," = w\n", h," = h\n")
		#plt.semilogx(w*f/2/np.pi, np.log10(abs(h)))
		#plt.title('bandpass Filter used')
		#plt.show()			
		#calc Fourier(data)*Fourier(filter)
		for i in range(np.size(data_fft,0)):
			for j in range(np.size(data_fft,1)):
				data_fft[i][j] = data_fft[i][j]*h[j]
		#ifft
		if not data_is_fourier:
			return np.fft.irfft(data_fft)
		else:
			return data_fft
	def integrate(self, data):
		L = self.object().property("MeasureLength[m]")
		dx = self.object().property("SpatialResolution[m]")
		T = self.object().property("StreamTime[s]")
		f = self.object().property("SamplingFrequency[Hz]")
		dt=1/f
		time_pts = np.linspace(0,dt,f*T)
		res = data.copy()
		for i in range(len(self.group_channels('Measurement'))):
			res[i][0]=0
			for j in range(int(f*T)-1):						
				res[i][j+1] = res[i][j]+integrate.simps(data[i][j:j+2], time_pts[j:j+2], dt)
		return res
	def get_timeAverage(self, data):
		res = np.zeros(np.size(data, 0))
		for i in range(np.size(data, 0)):
			res[i] = sum(data[i])/np.size(data, 1)
		return res
	def filter_moys(self, data, option="single_phase_flow"):
		data = signal.detrend(data, 0)
		data = signal.detrend(data, 1)
		return 
	#data :: axis 0:k axis 1:t
	def calc_2DFFT(self, data, distance_conversion_factor=1, fft_size_k=None, fft_size_t=None, stack_t=False, stack_t_interval=20000):
		samplingf = self.object().property("SamplingFrequency[Hz]")
		dx = self.object().property("SpatialResolution[m]")
		if stack_t == True:
			n = int(np.size(data, 1)/stack_t_interval)
			data_fft = np.zeros([np.size(data,0), int(stack_t_interval/2)+1], dtype=complex)
			for i in range(n):
				data_fft += np.fft.rfft2(data[0:np.size(data,0),stack_t_interval*i:stack_t_interval*(i+1)])/n
		else:
			data_fft = np.fft.rfft2(data)
		
		if type(fft_size_k)==int:
			data_fft = signal.resample(data_fft, fft_size_k, axis=0)
		if type(fft_size_t)==int:
			data_fft = signal.resample(data_fft, fft_size_t, axis=1)
		timestep = 1./samplingf		
		distance_step = dx/distance_conversion_factor
		freq_Hz = np.fft.rfftfreq(np.size(data_fft,1)*2-1,timestep)#f
		K_wave_num = np.fft.fftfreq(np.size(data_fft,0),distance_step)#k=1/lambda
		sort = K_wave_num.argsort()
		K_wave_num = K_wave_num[sort]
		data_fft = data_fft[sort]#trier comme K_wave_num
		return freq_Hz, K_wave_num, data_fft

	def radial_scan(self, freq_Hz, K_Wave_num, data, option="linear_scan", velocity_resolution=0.5, velocity_interval=[1300,1700]):
		#reordonner les data
		if option == "linear_scan":
			K_r_bordel = K_Wave_num[K_Wave_num>0]
			K_l_bordel = np.abs(K_Wave_num[K_Wave_num<0])
			K_right_sort = K_r_bordel.argsort()
			K_left_sort = K_l_bordel.argsort()
			K_r = np.sort(K_r_bordel)
			K_l = np.sort(K_l_bordel)
			#scan right
			vel_r = np.zeros(np.size(K_r))
			for i in range(np.size(K_r)):
				vel_r[i] = freq_Hz[np.argmax(np.abs(data[K_Wave_num>0][K_right_sort[i]]))]/K_r[i]			
			velmoy_right = sum(vel_r)/np.size(K_r)
			#scan left
			vel_l = np.zeros(np.size(K_l))
			for i in range(np.size(K_l)):
				vel_l[i] = freq_Hz[np.argmax(np.abs(data[K_Wave_num<0][K_left_sort[i]]))]/np.abs(K_l[i])
			velmoy_left = sum(vel_l)/np.size(K_l)
			#verif (optionel)
			plt.plot(K_r, vel_r)
			plt.plot(K_l, vel_l)
			plt.show()
			#fluid_vel_calc
			fluidvel = (velmoy_right-velmoy_left)/2
			return fluidvel, velmoy_left, velmoy_right, vel_l, vel_r
		elif option == "angular_scan":
			vels_r = np.outer(1/K_Wave_num[K_Wave_num>0],freq_Hz)
			vels_l = np.outer(-1/K_Wave_num[K_Wave_num<0],freq_Hz)
			v_test = np.linspace(np.min(velocity_interval), np.max(velocity_interval), int((np.max(velocity_interval)-np.min(velocity_interval))/velocity_resolution))
			amp_r, amp_l = np.zeros(np.size(v_test)), np.zeros(np.size(v_test))			
			for i in range(np.size(v_test)):		
				ind_r = np.abs(vels_r-v_test[i])<velocity_resolution/2
				ind_l = np.abs(vels_l-v_test[i])<velocity_resolution/2
				if np.size(data[K_Wave_num>0][ind_r])==0 or np.size(data[K_Wave_num<0][ind_l])==0:
					break #failure: velocity resolution too low
				amp_r[i] = sum(np.abs(data[K_Wave_num>0][ind_r]))/np.size(data[K_Wave_num>0][ind_r])
				amp_l[i] = sum(np.abs(data[K_Wave_num<0][ind_l]))/np.size(data[K_Wave_num<0][ind_l])
			#lissage
			#w_l = np.size(v_test)
			#if w_l%2 == 0:
			#	 w_l = np.size(v_test)-1
			#else:
			#	w_l = np.size(v_test)
			#amp_r = signal.savgol_filter(amp_r, w_l, 2)
			#amp_l = signal.savgol_filter(amp_l, w_l, 2)
			#verif (optionel)
			plt.plot(v_test, amp_r)
			plt.plot(v_test, amp_l)
			plt.show()
			vl, vr = v_test[np.argmax(amp_l)], v_test[np.argmax(amp_r)]
			return np.abs(vl-vr)/2, vl, vr, amp_l, amp_r
		elif option == "power_scan":
			vels_r = np.outer(1/K_Wave_num[K_Wave_num>0],freq_Hz).flatten()
			vels_l = np.outer(-1/K_Wave_num[K_Wave_num<0],freq_Hz).flatten()
			amp_r = np.abs(data[K_Wave_num>0]).flatten()
			amp_l = np.abs(data[K_Wave_num<0]).flatten()
			#calc vel fluid
			vf = (np.abs(sum(vels_r*amp_r)-sum(vels_l*amp_l))/(sum(amp_r)+sum(amp_l)))/2
			return vf
			

		
	def calc_gradient(self, freq_Hz, K_Wave_num, data, option="kf"):
		dk = np.abs(K_Wave_num[1]-K_Wave_num[0])
		df = np.abs(freq_Hz[1]-freq_Hz[0])
		return np.gradient(data, df, dk, edge_order=1, axis=None) 
	##################	
	##Plot functions##
	##################
	def plot3D_colors(self, option='raw', channel_start=0, channel_end=None, time_start=0, time_end=None):
		# Make data.
		channels, times = self.limits_to_arrays(channel_start, channel_end, time_start, time_end)
		X, Y = np.meshgrid(times, channels)
		data = self.load_data(channels, times)
		#plot data
		if option == 'raw':
			plt.title('raw data')
			#rien
		elif option == 'lowpassFilter':
			#application d'un filtre de Butterworth passe bas
			data = self.filter_lowpass(data)
			plt.title('data filtered by lowpass filter')
		elif option == 'timeIntegrated':
			#application d'un filter de Butterworth passe bas
			data = self.filter_lowpass(data)
			#integration par rapport au temps au temps
			data = self.integrate(data)
			plt.title('timeIntegrated data')

		print(np.size(X,0),np.size(X,1))
		print(np.size(Y,0),np.size(Y,1))
		print(np.size(data,0),np.size(data,1))

		plt.pcolor(X, Y, data.real, cmap='RdBu', vmin = data.real.min(), vmax = data.real.max())
		plt.axis([X.min(), X.max(), Y.min(), Y.max()])
		plt.colorbar()
	def plot_2DFFT_colors(self, channel_start=0, channel_end=None, time_start=0, time_end=None, distance_conversion_factor=1, stack_t=False, stack_t_interval=20000):
		# Make data.
		channels, times = self.limits_to_arrays(channel_start, channel_end, time_start, time_end)
		data = self.load_data(channels, times)
		freqs_t, freqs_space, data_fft = self.calc_2DFFT(data, distance_conversion_factor, None, None, stack_t, stack_t_interval)

		f, K = np.meshgrid(freqs_t, freqs_space)

		#plot data
		plt.pcolor(f, K, np.abs(data_fft), cmap='RdBu', vmin = 0, vmax = 1e6)
		plt.axis([f.min(), f.max(), K.min(), K.max()])
		plt.colorbar()
	#plot 3D data
	def plot3D (self):
		from mpl_toolkits.mplot3d import Axes3D
		from matplotlib import cm
		from matplotlib.ticker import LinearLocator, FormatStrFormatter
		
		fig = plt.figure()
		ax = fig.gca(projection='3d')

		# Make data.
		L = self.object().property("MeasureLength[m]")
		dx = self.object().property("SpatialResolution[m]")
		T = self.object().property("StreamTime[s]")
		f = self.object().property("SamplingFrequency[Hz]")

		xsize = f*T
		ysize = round(L/dx)
		X = np.linspace(0,T, xsize) #en sec
		Y = np.linspace(0,L, ysize)# en m de fibre
		X, Y = np.meshgrid(X, Y)
		data = self.load_data()
			
		# Plot the surface.
		surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm, linewidth=0, antialiased=False)

		# Customize the z axis.
		ax.set_zlim(np.amin(data), np.amax(data))
		ax.zaxis.set_major_locator(LinearLocator(11))

		# Add a color bar which maps values to colors.
		fig.colorbar(surf, shrink=0.5, aspect=5)
	
	def plot2D_timeAverage(self, channel_start=0, channel_end=None, time_start=0, time_end=None):
		channels, times = self.limits_to_arrays(channel_start, channel_end, time_start, time_end)
		data = self.load_data(channels, times)

		dx = self.object().property("SpatialResolution[m]")
		res = self.get_timeAverage(data)
		X = channels*dx
		plt.plot(X, res)
	#plot 2D data at loc in channel
	def plot2D_channel(self, channels, time_start=0, time_end=None):
		times = self.t_limits_to_array(time_start, time_end)
		X = times/self.object().property("SamplingFrequency[Hz]")
		data = self.load_data(channels, times)
		print(np.size(data,0), np.size(data, 1))
		plt.plot(X, data.transpose())
	def plot_time_FFT(self, channels, time_start=0, time_end=None):
		times = self.t_limits_to_array(time_start, time_end)
		data = self.load_data(channels, times)
		data_fft = np.fft.rfft(data.transpose())
		sample_rate = self.object().property("SamplingFrequency[Hz]")
		X = np.fft.rfftfreq(np.size(data_fft,0), 1./sample_rate)
		print(np.size(X),np.size(data_fft[0:int(np.size(data_fft,0)/2)+1]))
		plt.plot(X, data_fft[0:int(np.size(data_fft,0)/2)+1])

	def plots_show(self):
		plt.show()

class mass_posttraitement():
	def __init__(self, path):
		super().__init__()
		if (type(path) == str):
			self.path=path

	def plot2D_timeAverage(self, channel_start=0, channel_end=None, time_start=0, time_end=None):
		filepath = glob.glob(self.path+'*.tdms')
		for i in range(len(filepath)):
			d = tdms_data(filepath[i])
			d.plot2D_timeAverage(channel_start, channel_end, time_start, time_end)
		plt.show()
	def plot2D_compare_timeAveragemax(self, comp_with, option='lin', channel_start=0, channel_end=None, time_start=0, time_end=None):
		filepath = glob.glob(self.path+'*.tdms')
		res = np.zeros(len(filepath))
		for i in range(len(filepath)):
			string = filepath[i].replace('_',' ').split()
			j = int(string[1])
			d = tdms_data(filepath[i])
			channels, times = self.limits_to_arrays(channel_start, channel_end, time_start, time_end)
			res[j-1] = np.max(d.get_timeAverage(d.load_data(channels, times)))
		if option == 'lin':				
			plt.plot(comp_with, res)
		elif option == 'log10':
			plt.semilogx(comp_with,np.log10(res))		
		plt.show()
	def plot2D_velocities(self, channel_start=0, channel_end=None, time_start=0, time_end=None, option="angular_scan"):
		filepath = glob.glob(self.path+'*.tdms')
		res = np.zeros(len(filepath))
		for i in range(len(filepath)):
			string = filepath[i].replace('_',' ').split()
			j = int(string[1])
			print("file", j, string[2])
			d = tdms_data(filepath[i])
			print("load_data")
			channels, times = d.limits_to_arrays(channel_start, channel_end, time_start, time_end)
			data = d.load_data(channels, times)
			print("calc fft")
			f, K, d_fft = d.calc_2DFFT(data, distance_conversion_factor=116, fft_size_k=10000, stack_t=False, stack_t_interval=20000)
			print("radial scan")			
			vf = d.radial_scan(f, K, d_fft, option, velocity_resolution=0.5, velocity_interval=[1300,1700])
			res[j-2] = vf[0]
			print(res[j-2])
		plt.plot(res)
		plt.show()
		return res
		

			

			
		

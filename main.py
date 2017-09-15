from tdms_data import *
import numpy as np



datafolderpath = "data/water_1in/optical/"
filenameIn = datafolderpath+"10_170803154549.tdms"

d = tdms_data(filenameIn)
#d.read_data_info()

#Test plot 2DFFT
#d.plot_2DFFT_colors(channel_start=230 ,channel_end=340, time_start=0, time_end=20000, distance_conversion_factor=116, stack_t=False, stack_t_interval=20000)
#d.plots_show()

#Test radial scan#
channels, times = d.limits_to_arrays(230, 340, 0, 20000)
print("load data")
data = d.load_data(channels, times)
print("calc 2D FFT")
f, K, d_fft = d.calc_2DFFT(data, distance_conversion_factor=116, fft_size_k=10000, stack_t=False, stack_t_interval=20000)
print("radial scan")#
vf = d.radial_scan(f, K, d_fft, option="angular_scan", velocity_resolution=0.5, velocity_interval=[1300, 1700])
print("calc succes")
print(vf[0])
####################################################################################################
#FLODAS MASS POSTTRAITEMENT
#datafolderpath = "../../../FLODAS/dataEpernon/27072017/waterSolo1inch/"
#d = mass_posttraitement(datafolderpath)
#d.plot2D_velocities(channel_start=230 ,channel_end=340, time_start=0, time_end=20000)



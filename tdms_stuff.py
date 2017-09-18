from user_idas import DasData
import numpy as n
from matplotlib import pyplot as plt
from scipy import signal
import time

t = time.time()
# f = DasData("data/water_1in/optical/10_170803154549.tdms","temp")
f = DasData("data/water_10m3h_1inch_170803154549.tdms")
tt = time.time()
print("data loaded in " + str(tt-t) + "s")

res = f.fft_calc(230,330,0.1,1,"ortho")
# res2 = signal.resample(res,100,axis=1)
# freq = n.fft.rfftfreq
# res = res / c
print("FFT mean in ",str(time.time()-tt),"s")
res2 = signal.resample(res,100,axis=1)
plt.imshow(abs(res2))

plt.show()
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
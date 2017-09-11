from user_idas import DasData
import numpy as n
from matplotlib import pyplot as plt
import time

t = time.time()
f = DasData("data/water_1in/optical/10_170803154549.tdms","temp")
tt = time.time()
print("data loaded in " + str(tt-t) + "s")

Tmean = 0.5
Tend = 60
c = 0
dt = int(f.fsamp * Tmean)
res = n.zeros((100,dt))
res = n.fft.fft2(res)
for i in range(0,int(Tend*f.fsamp),dt):
	print(i)
	res += n.fft.fft2(f.get_array(10,110,i,int(i+dt)))
	c+= 1
res = res / c
print("FFT mean in ",str(time.time()-tt),"s")
# plt.imshow(abs(res))
# plt.show()
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
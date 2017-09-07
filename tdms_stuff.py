from user_tdms import UTdms
import numpy as n


f = UTdms("data/water_7m3h_1inch_170803153239.tdms")

fech = int(f.get_param("SamplingFrequency[Hz]"))

Tmean = 0.1
chs = f.get_channels(0)
Nl = len(chs)
Nt = len(f.get_channel(0,0).data)
Tmax = Nt/fech
res = n.zeros((Nl,int(Tmean*fech)))
ii = int(Tmean*fech)
for i in range(Nt,int(Tmean*fech)):
	print(n.size(res))
	for j in range(4):
		res[j,0:int(Tmean*fech)] = f.get_channel(0,j).data[i:ii]
	ii += int(Tmean*fech)
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
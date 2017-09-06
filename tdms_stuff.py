from nptdms import TdmsFile
import sys

tdms_file = TdmsFile("data/water_1in/optical/5_170803152312.tdms")
# Lecture des propriétés
params = tdms_file.object()
# for name, value in rooto.properties.items():
#     print("{0}: {1}".format(name, value))
# Liste des groupes
# print("\n","Liste des groupes :","\n",tdms_file.groups(),"\n")
# Liste des channels d'un groupe
# print("Liste des channels d'un groupe","\n",tdms_file.group_channels('Measurement'),"\n")
print(tdms_file.object())

channels = tdms_file.group_channels('Measurement')


# Liste des données d'un channel
# channel = tdms_file.object('Untitled','Untitled')
# data = channel.data
# time = channel.time_track()
# print("DATA","\n",data,"\n")
# print("TIME","\n",time,"\n")
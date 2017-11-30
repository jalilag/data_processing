import numpy as n
from scipy import signal
import pickle

def sort_vecs(data,vecs):
	data = n.array(data)
	for i in range(len(data)):
		if i < len(vecs) and vecs[i] is not None:
			xsort = vecs[i].argsort()
			vecs[i]=vecs[i][xsort]
			if i == 0: data = data[xsort]
			if i == 1: data = data[:,xsort]
			if i == 2: data = data[:,:,xsort]
	return data,vecs

def crop_vecs(data,vecs,crop):
	data = n.array(data)
	N = len(n.shape(data))
	for i in range(N):
		if crop[2*i] == None: crop[2*i] = n.min(vecs[i])
		if crop[2*i+1] == None: crop[2*i+1] = n.max(vecs[i])
		xsort = n.where((vecs[i] >= crop[2*i]) & (vecs[i] <= crop[2*i+1]))[0]
		vecs[i]=vecs[i][xsort]
		if i == 0: data = data[xsort]
		if i == 1: data = data[:,xsort]
		if i == 2: data = data[:,:,xsort]
	return data,vecs


def set_to_lim(data,min_lim = None,max_lim=None):
	data = n.array(data)
	if min_lim is not None: data[n.where(data<=min_lim[0])] = min_lim[1]
	if max_lim is not None: data[n.where(data>=max_lim[0])] = max_lim[1]
	return data

def resamp_vecs(data,vecs,N,sort_data=True):
	for i in range(len(N)):
		if N[i] is not None:
			data = signal.resample(data,N[i],axis=i)
			if vecs[i] is not None: vecs[i] = signal.resample(vecs[i],N[i])
	if sort_data: data,vecs = sort_vecs(data,vecs)
	return data,vecs

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def put_mat_in_vec(data):
	return n.reshape(data,int(n.shape(data)[0]*n.shape(data)[1]))

def normL3(res):
	return res/n.max(n.abs(res))
import pylab as pl
import numpy as np

nx = 2880
ny = 5760
shape = (nx,ny)
sz = nx*ny
data1 = np.random.rand(sz).reshape(shape)
data2 = np.random.rand(sz).reshape(shape)
data3 = data2-data1

pl.clf() ; pl.imshow(data1)
pl.clf() ; pl.imshow(data2)
pl.clf() ; pl.imshow(data3)
pl .show()
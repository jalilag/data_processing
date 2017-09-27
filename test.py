import time
import numpy
import pyfftw
import multiprocessing
import sys
nthread = multiprocessing.cpu_count()
print(nthread)
# sys.exit()

a = numpy.random.rand(2364,2756).astype('complex128')
"""
Uncomment below to use 32 bit floats, 
increasing the speed by a factor of 4
and remove the difference between the "builders" and "FFTW" methods
"""
#a = numpy.random.rand(2364,2756).astype('complex64')

start = time.time()
b1 = numpy.fft.fft2(a)
end1 = time.time() - start
print(end1)

# start = time.time()
# b2 = pyfftw.interfaces.scipy_fftpack.fft2(a, threads=nthread)
# end2 = time.time() - start

# pyfftw.forget_wisdom()
# start = time.time()
# b3 = pyfftw.interfaces.numpy_fft.fft2(a,flags=('FFTW_ESTIMATE', ), threads=nthread)
# end3 = time.time() - start
# print(end3)

""" By far the most efficient method """
pyfftw.forget_wisdom()
start = time.time()
b4 = numpy.zeros_like(a)
fft = pyfftw.FFTW( a, b4, axes=(0,1), direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE', ), threads=8, planning_timelimit=None )
fft()
end4 = time.time() - start
print(end4)

""" 
For large arrays avoiding the copy is very important, 
doing this I get a speedup of 2x compared to not using it 
"""
pyfftw.forget_wisdom()
start = time.time()
b5 = numpy.zeros_like(a)
fft = pyfftw.builders.fft2(a, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
b5 = fft()
# fft = pyfftw.builders.fft2(a, s=None, axes=(-2, -1), overwrite_input=False, planner_effort='FFTW_MEASURE', threads=nthread, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
# b5 = fft()
end5 = time.time() - start
print(end5)

pyfftw.forget_wisdom()
start = time.time()
b6 = numpy.zeros_like(a)
fft = pyfftw.interfaces.numpy_fft.fft2(a, s=None, axes=(-2, -1), norm=None, overwrite_input=False, planner_effort='FFTW_ESTIMATE', threads=8, auto_align_input=True, auto_contiguous=True)
print(time.time()-start)

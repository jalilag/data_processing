from multiprocessing import Pool,Process
from itertools import product

def f(a, b):
    return a*b

if __name__ == '__main__':
	p = Process(target=f, args=(4, 5))
	p.start()
	p.join()
	# print(results)
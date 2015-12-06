#!/usr/bin/env python
from multiprocessing import Pool
def myprint(x):
    return(x)
if __name__ == '__main__':
    p = Pool(2)
    print(p.map(myprint,range(100)))

'''
Created on Feb 26, 2014

@author: deschild
'''


import scipy.signal
import pylab
import numpy
from numpy import pi,exp,sqrt



sigmas=[0.25,0.5,1,2,4,8,16,32,64,128]
g=[]

def mexican_hat(a,sigma):
    x=numpy.arange(-a//2,a//2+1)
    return -sigma**2/(sigma*sqrt(2*pi))*(x**2/sigma**4-1/sigma**2)*exp(-x**2/(2*sigma**2))
#    return -(x**2/sigma**2-1)*exp(-x**2/(2*sigma**2))


m=[]
# m=mexican_hat(100,1)
# pylab.plt.plot(m)
# pylab.show()

for sigma in sigmas:
    w=10*sigma
    if w%2==0:w+=1
    c=(scipy.signal.gaussian(w, sigma))
    g.append(c)
    b=mexican_hat(w, sigma)
    m.append(b)

# pylab.plot(w,b)
raw=numpy.concatenate(g)
# # pylab.imshow((b,b))
# pylab.show()


DoGs = []
b = numpy.convolve(raw, g[0], mode="same")
for gaus in g[1:]:
    c = numpy.convolve(raw, gaus/gaus.sum(), mode="same")
    DoGs.append(b-c)
    b=c
LoGs = [ numpy.convolve(raw, mh, mode="same") for mh in m] 

f2 = pylab.figure(2)
sp1 = f2.add_subplot(211)

for s,crv in zip(sigmas,LoGs):
    sp1.plot(crv,label="$\sigma$=%s"%s)
sp2 = f2.add_subplot(212)
for s,crv in zip(sigmas,DoGs):
    sp2.plot(crv,label="$\sigma$=%s"%s)
sp1.plot(raw,label="raw")
sp2.plot(raw,label="raw")
sp1.legend()
sp2.legend()

f2.show()


raw_input("enter to quit")
'''
Created on Feb 24, 2014

@author: deschild
'''
import numpy
import scipy
from scipy import signal
from pylab import *
import fabio
from scipy import ndimage

ion()

blurs = []
kpx=[]
kpxs=[]
kpy=[]
kpys=[]
sigmas_abs=[]
sigmas_rel=[]

def convolve_gaussian(im,sigma):
        size =int( 8*sigma +1 )
        if size%2 == 0 :
            size += 1
        x = numpy.arange(size) - (size - 1.0) / 2.0
        gaus = numpy.exp(-x**2 / (sigma ** 2) / 2.0)
        gaus /= gaus.sum(dtype=numpy.float64)
        res2 = scipy.ndimage.filters.convolve1d(im, gaus, axis=0, mode= "reflect")
        res3 = scipy.ndimage.filters.convolve1d(res2, gaus, axis=1, mode= "reflect")
        return res3
    
if __name__ == '__main__':
    im = fabio.open("LaB6_0003.mar3450").data
#     sigma = [1.22627349847,1.54500779364,1.94658784146,2.45254699693,3.09001558729]
#     sigma=[1.6,1.96,2.47,3.11, 3.92]
    scales = 3
    sigma_init = 1.0
    sigma_prec = sigma_init
    sigma_final = 2.0
    sigma_ratio = sigma_final/sigma_init
#     sigma=[0.25,0.306,0.375,0.460,0.563,0.689,0.844,1.035,1.267,1.553,1.902,2.33,2.86,3.497,4.284]
#     sigma = numpy.arange(0.25,32+2**2,2**2)
#     tmp=convolve_gaussian(im,sigma_init)
#     blurs.append(tmp)
#     sigmas_abs.append(sigma_init)
#     sigmas_rel.append(sigma_init)
    tmp=im
    for scale in range(scales):
        sigma_abs = sigma_init*(sigma_ratio)**((scale)*1.0/scales)
        if scale == 0 :sigma_rel = sigma_prec
        else: sigma_rel = sigma_prec * numpy.sqrt(sigma_ratio**(2.0/scales)-1.0)
        print sigma_prec,sigma_abs,sigma_rel
        tmp = convolve_gaussian(tmp, sigma_rel)
        blurs.append(tmp)
        sigma_prec = sigma_abs
#         print scale
        sigmas_abs.append(sigma_abs)
        sigmas_rel.append(sigma_rel)
        
    print len(blurs)
    last_blur = blurs[0]
    dogs=numpy.empty((scales,im.shape[0],im.shape[1]))
    for i in range(len(blurs)-1):
        dogs[i]=(blurs[i]-blurs[i+1])
    kpms = []
    for last_s,cur_s,next_s in zip(dogs[0:],dogs[1:],dogs[2:]):
        kpm = numpy.zeros(shape=im.shape, dtype=numpy.uint8) 
        slic = cur_s[1:-1,1:-1]
        kpm[1:-1,1:-1] += (slic>cur_s[:-2,1:-1]) * (slic>cur_s[2:,1:-1])
        kpm[1:-1,1:-1] += (slic>cur_s[1:-1,:-2]) * (slic>cur_s[1:-1,2:])
        kpm[1:-1,1:-1] += (slic>cur_s[:-2,:-2])  * (slic>cur_s[2:,2:])
        kpm[1:-1,1:-1] += (slic>cur_s[2:,:-2])   * (slic>cur_s[:-2,2:])
        #kpm[1:-1,1:-1] += (slic>cur_s[1:-1,1:-1])

        #with next DoG
        kpm[1:-1,1:-1] += (slic>next_s[:-2,1:-1]) * (slic>next_s[2:,1:-1])
        kpm[1:-1,1:-1] += (slic>next_s[1:-1,:-2]) * (slic>next_s[1:-1,2:])
        kpm[1:-1,1:-1] += (slic>next_s[:-2,:-2])  * (slic>next_s[2:,2:])
        kpm[1:-1,1:-1] += (slic>next_s[2:,:-2])   * (slic>next_s[:-2,2:])
        kpm[1:-1,1:-1] += (slic>=next_s[1:-1,1:-1])

        #with previous DoG
        kpm[1:-1,1:-1] += (slic>last_s[:-2,1:-1]) * (slic>last_s[2:,1:-1])
        kpm[1:-1,1:-1] += (slic>last_s[1:-1,:-2]) * (slic>last_s[1:-1,2:])
        kpm[1:-1,1:-1] += (slic>last_s[:-2,:-2])  * (slic>last_s[2:,2:])
        kpm[1:-1,1:-1] += (slic>last_s[2:,:-2])   * (slic>last_s[:-2,2:])
        kpm[1:-1,1:-1] += (slic>=last_s[1:-1,1:-1])
        kpms.append(kpm)
    
    f = figure(1)
    ax = f.add_subplot(1,2,1)
    ax.imshow(numpy.log1p(im), interpolation="nearest")
    for i,kpm in enumerate(kpms):
        kpy,kpx=numpy.where(kpm >= 14)
        p = ax.plot(kpx,kpy,"o")
        p[0].set_label("sigma=%.3f"%sigmas_abs[i])
        print("color",p[0].get_color())
    ax.legend()
    ax2 = f.add_subplot(1,2,2)
    ax2.plot(sigmas_abs,dogs[:,2398,3087])
    ax2.plot(sigmas_abs,dogs[:,1604,1670],label="center")
    ax2.legend()
# print kpxs
    f.show()
        
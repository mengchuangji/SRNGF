# -*- coding: utf-8 -*-
import numpy as np
 #######################计算PSNR
def psnr(data_origin,reconstructed): 
        
        diff = reconstructed - data_origin
        mse = np.mean(np.square(diff))
        psnr = 10 * np.log10(1 * 1 / mse)#最大值就是1
        return psnr
        
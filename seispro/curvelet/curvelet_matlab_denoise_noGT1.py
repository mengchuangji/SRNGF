
from seispro.psnr import psnr
import numpy as np
import matplotlib.pyplot as plt
import segyio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import matlab.engine
import matlab
import math
import os
import scipy.io as io

def show(x,y,method):
    import matplotlib.pyplot as plt
    plt.figure(dpi=500,figsize=(6,10)) #(12,9)
    plt.subplot(131)
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray') #'gray' #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    # plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(132)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    # plt.title('denoised')
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('../../noise/shot_' + method + '_dn.mat'), {'data': y[:, :, np.newaxis]})

    plt.subplot(133)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    # io.savemat(('../../noise/shot_' + method + '_n.mat'), {'data': noise[:, :, np.newaxis]})
    # plt.title('removed noise')
    plt.tight_layout()
    plt.show()
def readsegy(data_dir, file):
    filename = os.path.join(data_dir, file)
    with segyio.open(filename, 'r', ignore_geometry=True) as f:
        f.mmap()
        # print binary header info
        print(f.bin)
        print(f.bin[segyio.BinField.Traces])
        # read headerword inline for trace 10
        print(f.header[10][segyio.TraceField.INLINE_3D])
        # Print inline and crossline axis
        print(f.xlines)
        print(f.ilines)
        # Extract header word for all traces
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
        data = np.asarray([np.copy(x) for x in f.trace[:trace_num]]).T
        x = data[:, :]
        f.close()
        return x

eng = matlab.engine.start_matlab()
# eng.triarea(nargout=0)

data_dir = 'E:\博士期间资料\田亚军\田博给的数据\\2021.3.27数据'
original = readsegy(data_dir, '03-MonoNoiAtten-16_DYN_L1901-s11857.sgy')[400:800, 0:224] #3000,224 #[340:468, 48:176]#[640:768, 48:176] [576:822, 16:198] # 2351*787
y = original
##################################
y_max=max(abs(original.max()),abs(original.min()))
noisy_data=y/y_max

import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
# First argument must be double, single, int8, uint8, int16, uint16, int32, uint32, or logical.
# denoise_data = eng.fx_mssa(matlab.double(noisy_data.tolist()),1.0,124.0,0.001,9,0)
# [DATA_f] = fx_decon(DATA,dt,lf,mu,flow,fhigh);
denoise_data = eng.seismic_curvelet_denoise(matlab.double(noisy_data.tolist()),0.05); #30/255
denoise_data=np.array(denoise_data)
print('haha')

noise_data=noisy_data-denoise_data
# from gain import *
# noisy_data=gain(noisy_data, 0.004, 'agc', 0.05, 1)
# denoise_data=gain(denoise_data, 0.004, 'agc', 0.05, 1)
show(noisy_data,denoise_data,method='curvelet')

# # wigb显示
from seis_util import wigb
x__ = noisy_data.copy()[72:400,70:90]  # parameter=0.05 [300:400,0:64]zoom
denoised__ = denoise_data.copy()[72:400,70:90]
noise__ = x__ - denoised__
x__max = abs(x__).max()

wigb.wigb(x__ / x__max, figsize=(3, 6), linewidth=1)  # (18, 30)(30, 18) (10, 6)zoom
wigb.wigb(denoised__ / x__max, figsize=(3, 6), linewidth=1)
wigb.wigb(noise__ / x__max, figsize=(3, 6), linewidth=1)



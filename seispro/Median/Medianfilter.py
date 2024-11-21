# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import segyio
from psnr import psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

def MedianFilter(src,  k = 3, padding = None):
#3x3尺寸的中值滤波来处理 	
    height, width = src.shape
    if not padding:
        edge = int((k-1)/2)
        if height - 1 - edge <= edge or width - 1 - edge <= edge:
            print("The parameter k is to large.")
            return None
        new_arr = np.zeros((height, width), dtype = "float32")
        for i in range(height):
            for j in range(width):
                if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
                    new_arr[i, j] = src[i, j]
                else:
                    new_arr[i, j] = np.median(src[i - edge:i + edge + 1, j - edge:j + edge + 1])

    return new_arr
		

def BetterMedianFilter(src, k = 3, padding = None):
    height, width = src.shape 
    if not padding:
        edge = int((k-1)/2)
        if height - 1 - edge <= edge or width - 1 - edge <= edge:
            print("The parameter k is to large.")
            return None
        new_arr = np.zeros((height, width), dtype = "float32")
        for i in range(height):
            for j in range(width):
                if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
                    new_arr[i, j] = src[i, j]
                else:
					#nm:neighbour matrix
                    nm = src[i - edge:i + edge + 1, j - edge:j + edge + 1]
                    max = np.max(nm)
                    min = np.min(nm)
                    if src[i, j] == max or src[i, j] == min:
                        new_arr[i, j] = np.median(nm)
                    else:
                        new_arr[i, j] = src[i, j]
    return new_arr
		

#############测试地震数据
# file_list1='../../seismic/test/00-L120-X.sgy' #'../data/noise.sgy'
file_list1='../../seismic/test/mix_test2-X.sgy'
f = segyio.open(file_list1, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T#(512,512)
noisy_data_max=abs(data).max()
noisy_data = data/noisy_data_max#归一化到-1,1之间

# file_list='../../seismic/test/00-L120-Y.sgy' #'../data/test1.sgy'
file_list='../../seismic/test/test2-Y.sgy'
f = segyio.open(file_list, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T#(512,512)
data_test = data/noisy_data_max#归一化到-1,1之间  #  mcj改了

########降噪处理

denoise_data=MedianFilter(noisy_data)#中值滤波
# denoise_data1=BetterMedianFilter(noisy_data)    #增强中值滤波
noise_data=noisy_data-denoise_data
     #########################画图地震数据图
 #########################画图地震数据图
clip = 1e-0#显示范围，负值越大越明显
vmin, vmax = -clip, clip
        # Figure
figsize=(12, 6)#设置图形的大小
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize, facecolor='w', edgecolor='k',
                               squeeze=False,sharex=True,dpi=100)
axs = axs.ravel()#将多维数组转换为一维数组
axs[0].imshow(data_test, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
axs[0].set_title('Clean')

axs[1].imshow(noisy_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# noisy_psnr = psnr(data_test,noisy_data)
psnr = compare_psnr(data_test.squeeze(), noisy_data.squeeze())
ssim = compare_ssim(data_test.squeeze(), noisy_data.squeeze())
# ssss=PSNR(data_test.squeeze(), noisy_data.squeeze())
# noisy_psnr=round(noisy_psnr, 2)
axs[1].set_title('Noisy\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr,ssim))

axs[2].imshow(denoise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# Denoised_psnr = psnr(data_test,denoise_data)
# Denoised_psnr=round(Denoised_psnr, 2)
psnr = compare_psnr(data_test.squeeze(), denoise_data.squeeze())
ssim = compare_ssim(data_test.squeeze(), denoise_data.squeeze())
axs[2].set_title('Denoised Median filter\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr,ssim))

axs[3].imshow(noise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# Denoised_psnr = psnr(data_test,denoise_data)
# Denoised_psnr=round(Denoised_psnr, 2)
axs[3].set_title('Noise Median filter')
plt.show()




import numpy as np
from pywt import dwt2, idwt2
import segyio
import matplotlib.pyplot as plt
from psnr import psnr

#############测试地震数据
file_list='../data/test1.sgy'
f = segyio.open(file_list, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T#(512,512)
data_test = data/abs(data).max()#归一化到-1,1之间

file_list1='../data/noise.sgy'
f = segyio.open(file_list1, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T#(512,512)
noisy_data = data#(288,288)
noisy_data = noisy_data/abs(noisy_data).max()#归一化到-1,1之间


"""小波降噪"""
coeffs2 = dwt2(noisy_data, 'haar')
LL, (LH, HL, HH) = coeffs2
# 根据小波系数重构回去的图像
denoise_data = idwt2((LL, (None, None, None)), 'haar')#只选择了有效的低频信号，其它3部分都是噪声
noise_data=noisy_data-denoise_data


 #########################画图地震数据图      
clip = 1e-0#显示范围，负值越大越明显
vmin, vmax = -clip, clip
        # Figure
figsize=(12, 3)#设置图形的大小
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize, facecolor='w', edgecolor='k',
                               squeeze=False,sharex=True,dpi=100)
axs = axs.ravel()#将多维数组转换为一维数组
axs[0].imshow(data_test, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
axs[0].set_title('Clear')

axs[1].imshow(noisy_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
noisy_psnr = psnr(data_test,noisy_data)
noisy_psnr=round(noisy_psnr, 2)
axs[1].set_title('Noisy, psnr='+ str(noisy_psnr))

axs[2].imshow(denoise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
Denoised_psnr = psnr(data_test,denoise_data)
Denoised_psnr=round(Denoised_psnr, 2)
axs[2].set_title('Denoised wavelet, psnr='+ str(Denoised_psnr))

axs[3].imshow(noise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
Denoised_psnr = psnr(data_test, noise_data)
Denoised_psnr = round(Denoised_psnr, 2)
axs[3].set_title('Noise wavelet, psnr=' + str(Denoised_psnr))
plt.show()

# -*- coding: utf-8 -*-
import segyio
from psnr import psnr
from seispro.mssa.mssa1 import mSSA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim


#############测试地震数据

# file_list1='../../seismic/test/00-L120-X.sgy' #'../data/noise.sgy'
file_list1='../../seismic/test/mix_test2-X.sgy'
f = segyio.open(file_list1, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T#[64,64]#(512,512)
noisy_data_max=abs(data).max()
noisy_data = data/noisy_data_max#归一化到-1,1之间

# file_list='../../seismic/test/00-L120-Y.sgy' #'../data/test1.sgy'
file_list='../../seismic/test/test2-Y.sgy'
f = segyio.open(file_list, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T#[64,64]#(512,512)
data_test = data/noisy_data_max#归一化到-1,1之间  #  mcj改了


noisy_data_pd = pd.DataFrame(noisy_data)#numpy转dataframe
 
##################MSSA处理
model = mSSA(rank_var=4)#建立模型
model.update_model(noisy_data_pd)#训练模型

df_column = np.zeros((data_test.shape[1],data_test.shape[0]))
for i, column in enumerate(noisy_data_pd.columns):
    df_more = model.predict(column,0,data_test.shape[0]-1)['Mean Predictions']
    df_column[i,:] = df_more
    
denoise_data=df_column.T
noise_data=noisy_data-denoise_data
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
axs[2].set_title('Denoised MSSA\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr,ssim))

axs[3].imshow(noise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# Denoised_psnr = psnr(data_test,denoise_data)
# Denoised_psnr=round(Denoised_psnr, 2)
axs[3].set_title('Noise MSSA')
plt.show()
#  #########################画图地震数据图
# clip = 1e-0#显示范围，负值越大越明显
# vmin, vmax = -clip, clip
#         # Figure
# figsize=(12, 3)#设置图形的大小
# fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize, facecolor='w', edgecolor='k',
#                                squeeze=False,sharex=True,dpi=100)
# axs = axs.ravel()#将多维数组转换为一维数组
# axs[0].imshow(data_test, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# axs[0].set_title('Clear')
#
# axs[1].imshow(noisy_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# noisy_psnr = psnr(data_test,noisy_data)
# noisy_psnr=round(noisy_psnr, 2)
# axs[1].set_title('Noisy, psnr='+ str(noisy_psnr))
#
# axs[2].imshow(denoise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# Denoised_psnr = psnr(data_test,denoise_data)
# Denoised_psnr=round(Denoised_psnr, 2)
# axs[2].set_title('Denoised MSSA, psnr='+ str(Denoised_psnr))
# noise_data=data_test-denoise_data
# axs[3].imshow(noise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# noised_psnr = psnr(data_test,denoise_data)
# noised_psnr=round(Denoised_psnr, 2)
# axs[3].set_title('noise MSSA')
# plt.show()

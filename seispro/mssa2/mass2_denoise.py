# -*- coding: utf-8 -*-
import segyio
from psnr import psnr
from seispro.mssa2.mssa22 import mSSA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#############测试地震数据
file_list='../../seismic/test/00-L120-Y.sgy'  #test-Y 00-L120
f = segyio.open(file_list, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T[:300,:60]#(512,512)
data_test = data/abs(data).max()#归一化到-1,1之间

file_list1='../../seismic/test/00-L120-X.sgy'
f = segyio.open(file_list1, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T[:300,:60]#(512,512)
noisy_data = data#(288,288)
noisy_data = noisy_data/abs(noisy_data).max()#归一化到-1,1之间


noisy_data_pd = pd.DataFrame(noisy_data)#numpy转dataframe
 
##################MSSA处理
model = mSSA()#建立模型
model.update_model(noisy_data_pd)#训练模型

df_column = np.zeros((data_test.shape[1],data_test.shape[0]))
for i, column in enumerate(noisy_data_pd.columns):
    df_more = model.predict(column,0,data_test.shape[0]-1)['Mean Predictions']
    df_column[i,:] = df_more
    
denoise_data=df_column.T

 #########################画图地震数据图      
clip = 1e-0#显示范围，负值越大越明显
vmin, vmax = -clip, clip
        # Figure
figsize=(10, 10)#设置图形的大小
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, facecolor='w', edgecolor='k',
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
axs[2].set_title('Denoised MSSA, psnr='+ str(Denoised_psnr))
plt.show()
######################画出去除的噪声
sub=noisy_data-denoise_data
clip = 1e-0#显示范围，负值越大越明显
vmin, vmax = -clip, clip
        # Figure
figsize=(3, 3)#设置图形的大小
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor='w', edgecolor='k',
                               squeeze=False,sharex=True,dpi=100)
axs = axs.ravel()#将多维数组转换为一维数组
axs[0].imshow(sub, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
axs[0].set_title('MSSA Noise')
plt.show()
#######################画出频率振幅谱
#获取干净信号第60道数据
clear=data_test.T[59]
noisy=noisy_data.T[59]
denoise=denoise_data.T[59]
Fs = 250;                 # 采样时间间隔为4毫秒，即每秒1/0.004=250个点采样率
n = len(clear)                  # 信号长度
k = np.arange(n)            #采样点数的等差数列k
T = n/Fs                    #共有多少个周期T
frq = k/T                  # 频率坐标
frq1 = frq[range(int(n/2))] # #由于对称性，取一半区间

clear_f = abs(np.fft.fft(clear))          # 振幅
clear_f1 = clear_f[range(int(n/2))]  ##画图曲线


noisy_f = abs(np.fft.fft(noisy))          # 振幅
noisy_f1 = noisy_f[range(int(n/2))]##画图曲线
noisy_f1 = clear_f1.max()/noisy_f1.max()*noisy_f1##规则化

denoise_f = abs(np.fft.fft(denoise))          # 振幅
denoise_f1 = denoise_f[range(int(n/2))]##画图曲线
denoise_f1 = clear_f1.max()/denoise_f1.max()*denoise_f1##规则化

#
#fig, ax = plt.subplots(1, 1)
#ax.plot(frq1,clear_f1,'r',label="Clean data") #绘制频谱
#ax.plot(frq1,noisy_f1,'b',label="Noisy data") #绘制频谱
#ax.set_xlabel('Frequency (Hz)')
#ax.set_ylabel('Amplitude')
#ax.legend()


fig, ax = plt.subplots(1, 1)
ax.plot(frq1,clear_f1,'r',label="Clean data") #绘制频谱
ax.plot(frq1,denoise_f1,'b',label="MSSA") #绘制频谱
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
ax.legend()
plt.show()

from seispro.psnr import psnr
import numpy as np
import matplotlib.pyplot as plt
import segyio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import matlab.engine
import matlab
import math
def compare_SNR(real_img,recov_img):
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))

    if noise_var ==0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var,10)
    return s
def show(x,y,x_):
    import matplotlib.pyplot as plt
    clip = abs(y).max()  # 显示范围，负值越大越明显
    vmin, vmax = -clip, clip
    plt.figure(figsize=(16, 30))  # 16,3
    plt.subplot(611)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)  # cmap='gray'
    plt.title('clean data')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(612)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('noisy data')
    # plt.colorbar(shrink= 0.5)

    # ####################
    snr_x_ = compare_SNR(x, y)
    print(' snr_before= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, y)
    print('psnr_before=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(ssim))
    ####################################################
    # ####################
    snr_x_ = compare_SNR(x, x_)
    print(' snr_after= {1:2.2f}dB'.format('test', snr_x_))
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_after=', '{:.4f}'.format(psnr_x_))
    ssim = compare_ssim(x, x_)
    print('ssim_after=', '{:.4f}'.format(ssim))
    ####################################################

    plt.subplot(613)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('denoised data')
    # plt.colorbar(shrink= 0.5)
    # io.savemat(('./noise/vdn_dn_uns_25.mat'), {'data': x_[:, :, np.newaxis]})

    plt.subplot(614)
    noise = y - x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('removed noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(615)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('groundtruth noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(616)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_ - x, vmin=vmin, vmax=vmax, cmap=plt.cm.seismic)
    plt.title('residual')
    # plt.colorbar(shrink=0.5)
    plt.show()
    # io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})

eng = matlab.engine.start_matlab()
# eng.triarea(nargout=0)

# file_list1='../../seismic/test/salt_35_N.sgy'
# f = segyio.open(file_list1, ignore_geometry=True)
# f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
# data = np.asarray([np.copy(x) for x in f.trace[:]]).T#(512,512)
# noisy_data_max=abs(data).max()
# noisy_data_max=1
# noisy_data = data/noisy_data_max#归一化到-1,1之间
#
# # file_list='../../seismic/test/00-L120-Y.sgy' #'../data/test1.sgy' test2-Y.sgy
# file_list='../../seismic/test/salt_35_Y.sgy'
# f = segyio.open(file_list, ignore_geometry=True)
# f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
# data = np.asarray([np.copy(x) for x in f.trace[:]]).T[:160, :640]#(512,512)
# data_test = data/noisy_data_max#归一化到-1,1之间  #  mcj改了

file_list1='../../seismic/field/00-L120-X.sgy' #'../data/test1.sgy' test2-Y.sgy
f = segyio.open(file_list1, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T[492:,0:480]#(512,512)
noisy_data_max=abs(data).max()
noisy_data = data/noisy_data_max#归一化到-1,1之间

file_list='../../seismic/field/00-L120-Y.sgy' #'../data/test1.sgy' test2-Y.sgy
f = segyio.open(file_list, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T[492:,0:480]#(512,512)
data_test = data/noisy_data_max#归一化到-1,1之间  #  mcj改了

import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
import time
start_time = time.time()
# First argument must be double, single, int8, uint8, int16, uint16, int32, uint32, or logical.
# denoise_data = eng.fx_mssa(matlab.double(noisy_data.tolist()),1.0,124.0,0.001,9,0)
# [DATA_f] = fx_decon(DATA,dt,lf,mu,flow,fhigh);
# %  IN   D:   	 intput 3D data
# %       flow:   processing frequency range (lower)
# %       fhigh:  processing frequency range (higher)
# %       dt:     temporal sampling interval
# %       N:      number of singular value to be preserved
# %       K:     damping factor
# %       verb:   verbosity flag (default: 0) D 0.0,120.0,0.004,64,1.0,0
denoise_data1 = eng.fxydmssa(matlab.double(noisy_data.tolist()),0.0,100.0,0.004,64,1.0,0);#0.004
noise_data1=noisy_data-np.array(denoise_data1)
# %  IN   signal:    initial signal
# %       noise:     initial noise
# %       rect:   3-D vector denoting smooth radius [20,20,1]
# %       niter:  number of CG iterations 100
# %       eps:    regularization parameter, default 0.0
# %       verb:   verbosity flag (default: 0)
denoise_data2,noise_data2,low=eng.localortho(matlab.double(denoise_data1),
                                         matlab.double(noise_data1.tolist()),
                                         matlab.double([[20, 20, 1]]),
                                         100,
                                         0.0,
                                         1, nargout=3) # 20,0,1
denoise_data=np.array(denoise_data2)
elapsed_time = time.time() - start_time
print(' %10s : %2.4f second' % ('lodmssa', elapsed_time))
print('haha')

noise_data=noisy_data-denoise_data
# clip = 1e-0#显示范围，负值越大越明显
# vmin, vmax = -clip, clip
#         # Figure
# figsize=(12, 6)#设置图形的大小
# fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize, facecolor='w', edgecolor='k',
#                                squeeze=False,sharex=True,dpi=100)
# axs = axs.ravel()#将多维数组转换为一维数组
# axs[0].imshow(data_test, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# axs[0].set_title('Clean')
#
# axs[1].imshow(noisy_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# # noisy_psnr = psnr(data_test,noisy_data)
# psnr = compare_psnr(data_test.squeeze(), noisy_data.squeeze())
# ssim = compare_ssim(data_test.squeeze(), noisy_data.squeeze())
# # ssss=PSNR(data_test.squeeze(), noisy_data.squeeze())
# # noisy_psnr=round(noisy_psnr, 2)
# axs[1].set_title('Noisy\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr,ssim))
#
# axs[2].imshow(denoise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# # Denoised_psnr = psnr(data_test,denoise_data)
# # Denoised_psnr=round(Denoised_psnr, 2)
# psnr = compare_psnr(data_test.squeeze(), denoise_data.squeeze())
# ssim = compare_ssim(data_test.squeeze(), denoise_data.squeeze())
# axs[2].set_title('Denoised MSSA\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr,ssim))
#
# axs[3].imshow(noise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# # Denoised_psnr = psnr(data_test,denoise_data)
# # Denoised_psnr=round(Denoised_psnr, 2)
# axs[3].set_title('Noise MSSA')
# plt.show()

show(x=data_test,y=noisy_data,x_=denoise_data)
from seis_util.plotfunction import show_DnNR_3x1, show_DnNR_1x3,show_DnNR_f_1x3_
# show_DnNR_3x1(x=data_test,y=noisy_data,x_=denoise_data)
# show_DnNR_1x3(x=data_test,y=noisy_data,x_=denoise_data,method='or+dmssa')
show_DnNR_f_1x3_(x=data_test,y=noisy_data,x_=denoise_data,method='or+dmssa')
from seis_util.plotfunction import show_DnNSimi_f_1x3_
from seis_util.localsimi import localsimi
simi = localsimi(noisy_data - denoise_data, denoise_data, rect=[5, 5, 1], niter=20, eps=0.0, verb=1)
energy_simi = np.sum(simi ** 2) / simi.size
print("energy_simi=", energy_simi)
show_DnNSimi_f_1x3_(x=data_test,y=noisy_data,x_=denoise_data,simi=simi.squeeze(),method='or+dmssa')
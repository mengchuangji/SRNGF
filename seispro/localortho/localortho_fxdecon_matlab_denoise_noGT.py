
# from seispro.psnr import psnr
import numpy as np
import matplotlib.pyplot as plt
import segyio
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import matlab.engine
import matlab
import math
import os
import scipy.io as sio

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
data_dir1 = 'E:\博士期间资料\田亚军\田博给的数据\盘客数据'
data_dir2 = 'E:\博士期间资料\田亚军\田博给的数据\盘客数据\\test\\'
# original = readsegy(data_dir, '03-MonoNoiAtten-16_DYN_L1901-s11857.sgy')[400:800, 0:224] #3000,224 #[340:468, 48:176]#[640:768, 48:176] [576:822, 16:198] # 2351*787
# original = readsegy(data_dir1, 'pk-00-L21-40-t400-4000.sgy')[:, 15902 - 796:15902][:1000, :500]
original = readsegy(data_dir2, 'PANKE-INline443.sgy')


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
import time
start_time = time.time()
denoise_data1 = eng.fx_decon(matlab.double(noisy_data.tolist()),matlab.double([0.002]),matlab.double([15]),matlab.double([0.01]),matlab.double([1]),matlab.double([124]));
noise_data1=noisy_data-np.array(denoise_data1)
denoise_data2,noise_data2,low=eng.localortho(matlab.double(denoise_data1),
                                         matlab.double(noise_data1.tolist()),
                                         matlab.double([[20, 20, 1]]),
                                         20,
                                         0.0,
                                         1, nargout=3) #np.asarray((20,20,1)).tolist()
denoise_data=np.array(denoise_data2)
elapsed_time = time.time() - start_time
print(' %10s : %2.4f second' % ('lofedecon', elapsed_time))
print('haha')

# sio.savemat(('../../output/results/20231129/PK443_orfx_dn.mat'), {'data': denoise_data[:, :]})
# sio.savemat(('../../output/results/20231129/PK443_orfx_n.mat'), {'data': (noisy_data-denoise_data)[:, :]})


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

show(noisy_data,denoise_data,method='lofxdecon')
# # wigb显示
# from seis_util import wigb
# x__ = noisy_data.copy()[72:400,70:90]  # parameter=0.05 [300:400,0:64]zoom
# denoised__ = denoise_data.copy()[72:400,70:90]
# noise__ = x__ - denoised__
# x__max = abs(x__).max()
#
# wigb.wigb(x__ / x__max, figsize=(3, 6), linewidth=1)  # (18, 30)(30, 18) (10, 6)zoom
# wigb.wigb(denoised__ / x__max, figsize=(3, 6), linewidth=1)
# wigb.wigb(noise__ / x__max, figsize=(3, 6), linewidth=1)

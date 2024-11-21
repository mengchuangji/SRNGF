import cv2

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

#############################
path = '..\..\seismic\\test\\'
clean_name = path + 'salt_35.sgy'
case = 2

f = segyio.open(clean_name, ignore_geometry=True)
f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
original = np.asarray([np.copy(x) for x in f.trace[:]]).T[:160, :640]  # (512,512)
H, W = original.shape
x = original
x_max = abs(x).max()  # 归一化到-1,1之间
x = x / abs(x).max()
# Generate the sigma map
from seis_util.generateSigmaMap import peaks, gaussian_kernel, sincos_kernel, generate_gauss_kernel_mix, \
    Panke100_228_19_147Sigma, MonoPao

if case == 1:
    # Test case 1
    sigma = peaks(256)
elif case == 2:
    # Test case 2
    sigma = sincos_kernel()
elif case == 3:
    # Test case 3
    sigma = generate_gauss_kernel_mix(256, 256)
elif case == 4:
    sigma = Panke100_228_19_147Sigma()
elif case == 5:
    sigma = MonoPao()
elif case == 6:
    sigma = gaussian_kernel()

sigma = 10 / 255.0 + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * ((100 - 10) / 255.0)
sigma = cv2.resize(sigma, (W, H))
# sigma_map = cv2.resize(generate_sigma(), (W, H))
np.random.seed(seed=0)  # for reproducibility

# # #######################
y = x + np.random.normal(0, 1, x.shape) * sigma[:, :]
# y = x + np.random.normal(0, 1, x.shape) * sigma_map
# y = x + np.random.normal(0, 30 / 255.0, x.shape)
# io.savemat(('./noise/noise_case3.mat'), {'data': y[:, :, np.newaxis]})
# y = loadmat('./noise/noise_case3.mat')['data'].squeeze()
# ################################
y_max=abs(y).max()
# y_max=1
y = y/y_max
x = x/y_max
snr_y = compare_SNR(x, y)
print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
psnr_y = compare_psnr(x, y)
print('psnr_y_before=', '{:.4f}'.format(psnr_y))
y_ssim = compare_ssim(x, y)
print('ssim_before=', '{:.4f}'.format(y_ssim))
noisy_data =y
data_test  = x

import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
# First argument must be double, single, int8, uint8, int16, uint16, int32, uint32, or logical.
from skimage.restoration import estimate_sigma
noise_level = float(estimate_sigma(noisy_data))
denoise_data = eng.seismic_curvelet_denoise(matlab.double(noisy_data.tolist()),100/255); #30/255 0.2941
denoise_data=np.array(denoise_data)
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
from seis_util.plotfunction import show_DnNR_3x1, show_DnNR_1x3
# show_DnNR_3x1(x=data_test,y=noisy_data,x_=denoise_data)
show_DnNR_1x3(x=data_test,y=noisy_data,x_=denoise_data,method='curvelet')
# import scipy.io as sio
# sio.savemat(('../../output/results/salt_sc100_' + 'curvelet' + '_dn.mat'),
#                     {'data': denoise_data[:, :]})
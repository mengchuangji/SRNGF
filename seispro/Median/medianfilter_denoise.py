
from seispro.psnr import psnr
import numpy as np
import matplotlib.pyplot as plt
import segyio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import matlab.engine
import matlab
import math
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

file_list1='../../seismic/test/salt_35_N.sgy'
f = segyio.open(file_list1, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T#(512,512)
noisy_data_max=abs(data).max()
noisy_data_max=1
noisy_data = data/noisy_data_max#归一化到-1,1之间

# file_list='../../seismic/test/00-L120-Y.sgy' #'../data/test1.sgy' test2-Y.sgy
file_list='../../seismic/test/salt_35_Y.sgy'
f = segyio.open(file_list, ignore_geometry=True)
f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
data = np.asarray([np.copy(x) for x in f.trace[:]]).T[:160, :640]#(512,512)
data_test = data/noisy_data_max#归一化到-1,1之间  #  mcj改了

# denoise_data=MedianFilter(noisy_data)
import cv2
denoise_data=cv2.medianBlur(noisy_data,5)

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
show_DnNR_1x3(x=data_test,y=noisy_data,x_=denoise_data,method='mf')
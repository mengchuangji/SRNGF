# -*- coding: utf-8 -*-

import argparse
import random
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from scipy.io import loadmat
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from skimage.measure import compare_ssim
import scipy.io as io

from seis_util.get_patch import *
import segyio
from seis_util.gain import *
from networks import VDN
from seis_util.utils import load_state_dict_cpu, peaks, sincos_kernel, generate_gauss_kernel_mix, batch_SSIM
import math

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)



def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))
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
def show(x,y,x_,n,x_max,sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,3))
    plt.subplot(171)
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('noised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(172)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray')
    plt.title('label')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(173)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(x_,vmin=-1,vmax=1,cmap='gray')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(174)
    noise= y-x_
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    plt.title('noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(175)
    n_real=n
    n_=y - x
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y-x,vmin=-1,vmax=1,cmap='gray')
    plt.title('groundtruth noise')
    # plt.colorbar(shrink=0.5)


    plt.subplot(176)
    residual= x_-x
    plt.imshow(residual, vmin=-1,vmax=1,cmap='gray')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('residual')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(177)
    plt.imshow(np.sqrt(sigma2))
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma')
    plt.colorbar(shrink= 0.8)
    plt.show()

def showsigma(sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,9))

    plt.imshow(np.sqrt(sigma2))#, vmin=0,vmax=1
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma')
    plt.colorbar(shrink= 0.8)
    plt.show()
    # print("sigma.median:",np.median(np.sqrt(sigma2)))

def showm(m):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,9))

    plt.imshow(m)#, vmin=0,vmax=1
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('m')
    plt.colorbar(shrink= 0.8)
    plt.show()

def show_gain(x, y, x_, n, sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 3))
    plt.subplot(161)
    y_gain = gain(y, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('noised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(162)
    x_gain = gain(x, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(163)
    x__gain = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x__gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(164)
    noise = y - x_
    noise_gain = gain(noise, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('noise')
    # io.savemat(('./noise/vdnnseming.mat'), {'data': noise_gain[:, :, np.newaxis]})
    # plt.colorbar(shrink=0.5)

    plt.subplot(165)
    n_real = n
    n_ = y - x
    n__gain = gain(n_, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(n__gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('groundtruth noise')
    # io.savemat(('./noise/702nog.mat'), {'data': n__gain[:, :, np.newaxis]})
    # plt.colorbar(shrink=0.5)

    plt.subplot(166)
    residual = x_ - x
    residual_gain = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(residual_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('residual')
    # plt.colorbar(shrink= 0.5)
    plt.show()

def NormMinandMax(npdarr, min=0, max=1):
    """"
    将数据npdarr 归一化到[min,max]区间的方法
    返回 副本
    """
    arr = npdarr.flatten()
    Ymax = np.max(arr)  # 计算最大值
    Ymin = np.min(arr)  # 计算最小值
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (npdarr - Ymin)

    return last


def readsegy(data_dir, file):
    filename = os.path.join(data_dir, file)
    f = segyio.open(filename, ignore_geometry=True)
    f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
    data = np.asarray([np.copy(x) for x in f.trace[:]]).T
    return data






class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

def plot_3_300(x):
    plt.figure(dpi=300, figsize=(3, 3))
    plt.imshow(x, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.axis('off')
    plt.show()
def plot_3_300_jet(x):
    plt.figure(dpi=300, figsize=(3, 3))
    plt.imshow(x, vmin=0, vmax=1, cmap=plt.cm.jet)
    plt.axis('off')
    plt.show()

use_gpu = True
case = 5
C = 1
dep_U = 4
# # clip bound
log_max = math.log(1e4)
log_min = math.log(1e-8)

if __name__ == '__main__':


    torch.set_default_dtype(torch.float32)
    # load the pretrained model
    print('Loading the Model')
    # checkpoint = torch.load('./model_state/model_state_seismic_v0_3') #c2 150 8.00
    # checkpoint = torch.load('./models_denoise/VDN_40-40/model_state_50') #c2 150 8.19
    # checkpoint = torch.load('./models_denoise/VDN_128-128/model_state_20')# Good  c2 150 9.11
    # checkpoint = torch.load('./models_denoise/fielddata/model_state_50')

      # depth=5
    # checkpoint = torch.load('./models_denoise/VDN_D5_32/model_state_50') #c2 150 8.28
      #VDN_NestedUNet
    # checkpoint = torch.load('./model_state/model_state_seismic_v1_518') #c2 150 9.33
    # checkpoint = torch.load('./models_denoise/VDN_NestedUNet_32-32/model_state_50')#c2 150 9.95
    # checkpoint = torch.load('./models_denoise/VDN_NestedUNet_48-48/model_state_20')  # c2 150 9.95
    # checkpoint = torch.load('./models_denoise/fielddata/VDN5p/model_state_1')
    # checkpoint = torch.load('./models_denoise/fielddata/semiSRNA/vdn/5_1_default/model_state_1')
    # checkpoint = torch.load('./models_denoise/fielddata/semiSRNA/vdn/model_state_23')

    # checkpoint = torch.load('./models_denoise/1111/fre/f_50/model_state_30')
    # checkpoint = torch.load('./models_denoise/fielddata/simu/VDN32/model_state_20')
    # checkpoint = torch.load('./models_denoise/1111/fre/f7/model_state_20')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/11XJVUn/model_state_1')  # 8最高8.76
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/2VUn/model_state_10')# 8最高8.76

    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/4VDn2/model_state_15')  # 15最高8.52/0.8110 DnCNN
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/4VDn3/model_state_3')  # 3最高8.47/0.8088
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/2VUn4/model_state_15')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/12VNUn/model_state_8')

    checkpoint = torch.load('../../model_zoo/field/VI-Non-IID-Unet/model_state_10')

    model = VDN(C, dep_U=dep_U, wf=64)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(model, checkpoint)
    model.eval()

    snrs = []

    data_dir='data/fielddata'
    im = '00-L120.sgy'

    data_dir = '../../seismic/field/'  #
    im = '00-L120.sgy'
    original = readsegy(data_dir, '00-L120-X.sgy')[492:, 0:480]  # [0:64,192:256]
    groundtruth = readsegy(data_dir, '00-L120-Y.sgy')[492:, 0:480]  # [0:64,192:256]
    noise = readsegy(data_dir, '00-L120-N.sgy')[492:, 0:480]  # [0:64,192:256]

    # AA = readsegy('data/fielddata/train/original', '00.sgy')
#############
    x=original
    H, W = x.shape
    if H % 2 ** dep_U != 0:
         H -= H % 2 ** dep_U
    if W % 2 ** dep_U != 0:
        W -= W % 2 ** dep_U
    x = x[:H, :W, ]
    groundtruth=groundtruth[:H, :W, ]
    noise=noise[:H, :W, ]
    #############################
    # x_max=x.max()
    x_max=max(abs(original.max()),abs(original.min()))
    x=x/x_max
    # io.savemat(('./noise/XJ-noisy.mat'), {'data': x[:, :]})
    groundtruth=groundtruth/x_max
    # io.savemat(('./noise/XJ-gt.mat'), {'data': x[:, :]})
    noise=noise/x_max
    # io.savemat(('./noise/702no.mat'), {'data': noise[:, :, np.newaxis]})
    ##############################
    #zoom
    x=x[32:96,:64]
    groundtruth=groundtruth[32:96,:64]
    noise=noise[32:96,:64]

    from skimage.measure import compare_psnr
    psnr_y = compare_psnr(groundtruth, x)
    print(' psnr_y_before= {1:2.2f}dB'.format('test', psnr_y))
    snr_y = compare_SNR(groundtruth, x)
    print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
    y_ssim = compare_ssim(groundtruth, x)
    print('ssim_before=', '{:.4f}'.format(y_ssim))
    ##################################
    x_ = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])
    torch.cuda.synchronize()
    start_time = time.time()
    if use_gpu:
        x_ = x_.cuda()
        print('Begin Testing on GPU')
    else:
        print('Begin Testing on CPU')
    with torch.autograd.set_grad_enabled(False):
        phi_Z = model(x_, 'test')
        err = phi_Z.cpu().numpy()
        phi_sigma = model(x_, 'sigma')
        phi_sigma.clamp_(min=log_min, max=log_max)
        # phi_sigma=phi_sigma#/phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)
        sigma2 = beta / (alpha + 1)
        sigma2 = sigma2.cpu().numpy()
        # io.savemat(('./noise/XJ-sigma-vdn-ul.mat'), {'data': np.squeeze(np.sqrt(sigma2))})
        sigma = np.sqrt(sigma2)
        print("sigma2.min:", sigma2.min(), "sigma2.median:", np.median(sigma), "sigma2.max:", sigma.max())

        m2 = np.exp(err[:, C:, ])  # variance
        m = np.sqrt(m2.squeeze())
        # io.savemat(('./noise/XJ-m-vdn-ul.mat'), {'data': m})
        print("m.min:", m.min(), "m.median:", np.median(m), "m.max:", m.max())

    if use_gpu:
        x_ = x_.cpu().numpy()
    else:
        x_ = x_.numpy()
    denoised = x_ - err[:, :C, ]
    denoised = denoised.squeeze()
    no = err[:, :C, ].squeeze()
    # io.savemat(('./noise/vnun-ul-n.mat'), {'data': no[:, :]})
    # io.savemat(('./noise/vunn-ul-dn.mat'), {'data': denoised[:, :]})


    # from scipy import stats
    # conf_intveral = stats.norm.interval(0.4, loc=denoised, scale=m)
    # from test_pro import low2upProbality
    # probality=np.zeros(conf_intveral[0].shape)
    # for i in range(0,conf_intveral[0].shape[0]):
    #     for j in range(0, conf_intveral[0].shape[1]):
    #         probality[i,j]=low2upProbality(L=(conf_intveral[0][i][j]-denoised[i][j])/m[i][j], U=(conf_intveral[1][i][j]-denoised[i][j])/m[i][j])
    # showm(probality)

    sigma2 = sigma2.squeeze()
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))
    snr_x_ = compare_SNR(groundtruth, denoised)
    psnr_x_ = compare_psnr(groundtruth, denoised)
    print('psnr_y_after=', '{:.4f}'.format(psnr_x_))
    # io.savemat(('./noise/702ori.mat'), {'data': groundtruth[:, :, np.newaxis]})
    # io.savemat(('./noise/702noise.mat'), {'data': x[:, :, np.newaxis]})


    ####################################################
    ssim = compare_ssim(groundtruth, denoised)
    print('ssim_after=','{:.4f}'.format(ssim) )
    ################################################


    if True:
        # name, ext = os.path.splitext(im)
        show(groundtruth, x, denoised,noise,x_max,sigma2)
        show_gain(groundtruth, x, denoised, noise, sigma2)
        from seis_util.plotfunction import show_DnNR_3x1, show_DnNR_1x3, show_DnNR_f_1x3_,show_DnNSimi_f_1x3_
        # show_DnNR_3x1(x=data_test,y=noisy_data,x_=denoise_data)
        # show_DnNR_1x3(x=data_test,y=noisy_data,x_=denoise_data,method='or+dmssa')
        show_DnNR_f_1x3_(x=groundtruth, y=x, x_=denoised, method='VI-Non-IID(Unet)')

        from seis_util.localsimi import localsimi
        simi = localsimi(x-denoised, denoised, rect=[5, 5, 1], niter=20, eps=0.0, verb=1)
        energy_simi = np.sum(simi ** 2)/simi.size
        print("energy_simi=", energy_simi)
        show_DnNSimi_f_1x3_(x=groundtruth, y=x, x_=denoised,simi=simi.squeeze(), method='MSE(Unet)')
        plot_3_300(denoised)
        plot_3_300(x- denoised)
        plot_3_300_jet(simi.squeeze())



        # showsigma(sigma2)
        # showm(m)
        # show_gain(groundtruth, x, denoised,noise,sigma2)#mcj

        #20221116
        # wigb显示
        from seis_util import wigb
        x__ = x.copy()[30:94, 0:64]  # parameter=0.05
        denoised__ = denoised.copy()[30:94, 0:64]
        noise__ = x__ - denoised__
        x__max=abs(x__).max()
        x__max=1
        wigb.wigb(x__/x__max, figsize=(10, 6), linewidth=1) #(18, 30)(30, 18)
        wigb.wigb(denoised__/x__max, figsize=(10, 6), linewidth=1)
        wigb.wigb(noise__/x__max, figsize=(10, 6), linewidth=1)







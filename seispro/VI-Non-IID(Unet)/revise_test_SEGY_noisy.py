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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fielddata', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=75, type=float, help='noise level')
    parser.add_argument('--agc', default=False, type=bool, help='Agc operation of the data,True or False')
    parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_sigma50'), help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results_denoise', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


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
def show(x,y,sigma2,x_max):
    clip = abs(x).max()
    vmin, vmax = -clip, clip
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,10)) #(12,9)
    plt.subplot(141)
    plt.imshow(x,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic) #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(142)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic)
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(143)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic)
    plt.title('removed noise')


    plt.subplot(144)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(sigma2)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('predicted sigma')
    plt.colorbar(shrink=0.5)
    plt.tight_layout()
    plt.show()
def showsigma(sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,9))

    plt.imshow(sigma2, vmin=0,vmax=1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma2')
    plt.colorbar(shrink= 0.8)
    plt.show()


def readsegy_ith_agc(data_dir, file,j,trace_per_shot,agc):
        filename = os.path.join(data_dir, file)
        with segyio.open(filename, 'r', ignore_geometry=True) as f:
            f.mmap()
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
            shot_num = int(float(trace_num / trace_per_shot))# 224 787
            len_shot = trace_num // shot_num  # The length of the data in each shot data
            data = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
            if agc:
                data = gain(data, 0.004, 'agc', 0.05, 1)
            # data = data/data.max()
            # data = data  # 先不做归一化处理
            x = data[:, :]
            f.close()
            return x
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

        # # Scatter plot sources and receivers color-coded on their number
        # plt.figure()
        # sourceY = f.attributes(segyio.TraceField.SourceY)[:]
        # nsum = f.attributes(segyio.TraceField.NSummedTraces)[:]
        # plt.scatter(sourceX, sourceY, c=nsum, edgecolor='none')
        #
        # groupX = f.attributes(segyio.TraceField.GroupX)[:]
        # groupY = f.attributes(segyio.TraceField.GroupY)[:]
        # nstack = f.attributes(segyio.TraceField.NStackedTraces)[:]
        # plt.scatter(groupX, groupY, c=nstack, edgecolor='none')

        trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
        data = np.asarray([np.copy(x) for x in f.trace[:trace_num]]).T
        x = data[:, :]
        f.close()
        return x

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

use_gpu = True
C = 1
dep_U = 4
# # clip bound
log_max = math.log(1e4)
log_min = math.log(1e-8)


if __name__ == '__main__':

    args = parse_args()
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

    # checkpoint = torch.load('./models_denoise/fielddata/simu/VDNN32/model_state_21')
    # checkpoint = torch.load('./models_denoise/fielddata/simu/VDN32/model_state_20')


    # checkpoint = torch.load('./models_denoise/fielddata/VDN5p/model_state_1')
    # checkpoint = torch.load('./models_denoise/fielddata/semiSRNA/vdnn/20210901/model_state_21')

    # print(torch.__version__)# 1.2.0原来的版本

    # checkpoint = torch.load('./models_denoise/1111/fre/f7/model_state_20')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/model_state_8')

    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/4VDn3/model_state_3') #7最好
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/2VUn3/model_state_10')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/12VNUn/model_state_20')


    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/11XJVUn/model_state_1')
    # checkpoint = torch.load('./models_denoise/1111/vdn_unet_75_100000/model_state_34')
    # checkpoint = torch.load('./models_denoise/1111field/vdn_unet_75_360000/model_state_1')

    # revise
    # 效果最好
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/13reviseVUn/eps1e8sgm75/model_state_50')#eps1e8sgm75
    #
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/13reviseVUn/eps1e9sgm75/model_state_1')
    #huangtuyuan
    # checkpoint = torch.load('./models_denoise/huangtuyuan/VUn/model_state_7')
    # VNUn
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/14reviseVNUneps8/model_state_50')

    # SRNGF
    checkpoint = torch.load('../../model_zoo/field/VI-Non-IID-Unet/model_state_10')

    model = VDN(C, dep_U=dep_U, wf=64)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(model, checkpoint)
    model.eval()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir, set_cur)

    data_dir='E:\博士期间资料\田亚军\田博给的数据\\2021.3.27数据'
    data_dir2='E:\博士期间资料\田亚军\田博给的数据\\20220128'
    data_dir3='E:\dataset\seismic'
    im = 'PANKE-INline443'
    # im = '03-MonoNoiAtten-16_DYN_L1901-s11857.sgy'
    # original=readsegy(data_dir,'PANKE-INline443.sgy',0,agc=False)[0:1600,0:768]#[0:800,0:768]##[100:228,19:147]#[0:1600,0:768]#[2000:2128,19:147]  #2351*787 [100:228,19:147] [1000:1128,19:147] [2000:2128,19:147]
    original = readsegy(data_dir, '03-MonoNoiAtten-16_DYN_L1901-s11857.sgy')[400:800, 0:224]
    # original = readsegy_ith_agc(data_dir2, 'BGP_00_huangtuyuan_pao.sgy', 19, trace_per_shot=224, agc=False)[224:624, 0:224]  #
    x = original

    #revise
    # original = readsegy_ith_agc(data_dir3, 'shots0001_0200.segy', 1, trace_per_shot=1201, agc=False)[:960,600:1193]#2001*1201
    # original=original/abs(original).max()
    # x=original+np.random.normal(0, 0.1, original.shape)
    x=x.astype(np.float32)
    import matplotlib.pyplot as plt
    clip = abs(original).max()
    vmin, vmax = -clip, clip
    plt.figure(figsize=(9, 10))  # (12,9)
    plt.imshow(original, vmin=vmin, vmax=vmax, cmap='gray')  # plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('clean')
    plt.show()

    H, W = x.shape
    if H % 2 ** dep_U != 0:
         H -= H % 2 ** dep_U
    if W % 2 ** dep_U != 0:
        W -= W % 2 ** dep_U
    x = x[:H, :W, ]

    #############################
    x_max=max(abs(original.max()),abs(original.min()))
    x=x/x_max
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
        phi_sigma=phi_sigma#/phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)
        sigma2 = beta / (alpha + 1)
        sigma2 = sigma2.cpu().numpy()
        # io.savemat(('./noise/PK-sigma-vdn-l.mat'), {'data': np.squeeze(np.sqrt(sigma2))})
        sigma=np.sqrt(sigma2)
        print("sigma.min:",sigma2.min(),"sigma.median:",np.median(sigma),"sigma.max:",sigma.max())
    if use_gpu:
        x_ = x_.cpu().numpy()
    else:
        x_ = x_.numpy()
    denoised = x_ - err[:, :C, ]

    m2=np.exp(err[:, C:,])  # variance
    m=np.sqrt(m2.squeeze())
    # io.savemat(('./noise/PK-m-vun-ul.mat'), {'data': m})
    print("m.min:", m.min(), "m.median:", np.median(m), "m.max:", m.max())


    denoised = denoised.squeeze()
    sigma2 = sigma2.squeeze()
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))
    no=err[:, :C, ].squeeze()

    # io.savemat(('./noise/PK-vnun-ul-n.mat'), {'data': no[:, :]})
    # io.savemat(('./noise/PK-vnun-ul-dn.mat'), {'data': denoised[:, :]})

    # io.savemat(('./noise/mn-vnun-ul-n.mat'), {'data': no[:, :]})
    # io.savemat(('./noise/mn-vnun-ul-dn.mat'), {'data': denoised[:, :]})

    if args.save_result:
        # name, ext = os.path.splitext(im)
        # x = gain(x, 0.004, 'agc', 0.05, 1)
        # denoised = gain(denoised, 0.004, 'agc', 0.05, 1)

        show(x,denoised,np.sqrt(sigma2),x_max)
        # showsigma(sigma2)


        # from seis_util.localsimi import localsimi
        # simi = localsimi(x-denoised, denoised, rect=[5, 5, 1], niter=20, eps=0.0, verb=1)
        # energy_simi = np.sum(simi ** 2)/simi.size
        # print("energy_simi=", energy_simi)


        # from seis_util.plotfunction import show_xynss,show_yxnSigma_gain,show_yxn_gain
        # show_xynss(x,denoised,np.sqrt(sigma2), simi.squeeze())
        # # show_yxnSigma_gain(x, denoised, np.sqrt(sigma2), agc=False)
        # show_yxn_gain(x, denoised, agc=False,method='vun')

        # # wigb显示
        from seis_util import wigb
        x__ = x.copy()[72:400, 70:90]  # parameter=0.05 [300:400,0:64]zoom
        denoised__ = denoised.copy()[72:400, 70:90]
        noise__ = x__ - denoised__
        x__max = abs(x__).max()

        wigb.wigb(x__ / x__max, figsize=(3, 6), linewidth=1)  # (18, 30)(30, 18) (10, 6)zoom
        wigb.wigb(denoised__ / x__max, figsize=(3, 6), linewidth=1)
        wigb.wigb(noise__ / x__max, figsize=(3, 6), linewidth=1)

    # if args.save_result:
    #     save_result(snrs, path=os.path.join(args.result_dir,'results.txt'))
    # log('Datset: {0:10s} \n  SNR = {1:2.2f}dB'.format('test', snr_avg))









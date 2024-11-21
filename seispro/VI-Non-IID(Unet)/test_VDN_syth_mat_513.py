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
from datasets.DenoisingDatasets_seismic_513 import gaussian_kernel
import scipy.io as io

from get_patch import *
import segyio
from gain import *
from networks import VDN
from utils import load_state_dict_cpu, peaks, sincos_kernel, generate_gauss_kernel_mix, batch_SSIM
from skimage.measure import compare_psnr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/test', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=float, help='noise level')
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

def show(x,y,x_):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,3))
    plt.subplot(161)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray')
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(162)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    plt.title('noised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(163)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_,vmin=-1,vmax=1,cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)
    io.savemat(('./noise/vdn_dn_uns_25.mat'), {'data': x_[:, :, np.newaxis]})


    plt.subplot(164)
    noise=y-x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise, vmin=-1, vmax=1,cmap='gray')
    plt.title('noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(165)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y-x, vmin=-1, vmax=1,cmap='gray')
    plt.title('groundtruth noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(166)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_-x, vmin=-1, vmax=1,cmap='gray')
    plt.title('residual')
    # plt.colorbar(shrink=0.5)
    plt.show()
    io.savemat(('./noise/vdn_res_uns_25.mat'), {'data': (x_ - x)[:, :, np.newaxis]})
def showsigma(sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,9))

    plt.imshow(sigma2)# vmin=0,vmax=1
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma2')
    plt.colorbar(shrink= 0.8)
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
def generate_sigma():
        pch_size = 128
        # center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        center = [pch_size/2, pch_size/2]
        # scale = random.uniform(pch_size/4*1, pch_size/4*3)
        scale = pch_size/4*2
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(10/255.0, 75/255.0)
        down = random.uniform(10/255.0, 75/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min())  *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :]
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
def MonoPao():
    zz = loadmat("E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\MonoPaoNoise\MonoPaoSigma.mat")['data']
    zz = np.sqrt(zz)
    # print(np.mean(zz))
    print("MonoPao.median",np.median(zz))
    print("MonoPao.max",zz.max())
    print("MonoPao.min",zz.min())
    return zz

def Gauss(sigma):
    sigma0=sigma/255
    sigma1=np.ones((256,256))*sigma0
    return sigma1

def Panke100_228_19_147Sigma():
    zz = loadmat("E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\MonoPaoNoise\PankeSigma100_228_19_147.mat")['data']
    zz = np.sqrt(zz)
    print("Panke100_228_19_147",np.median(zz))
    print("Panke100_228_19_147",zz.max())
    print("Panke100_228_19_147",zz.min())
    return zz
def gaussian_kernel():
    H = 128
    W = 128
    center = [64, 64]
    scale = 32
    centerH = center[0]
    centerW = center[1]
    XX, YY = np.meshgrid(np.arange(W), np.arange(H))
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerH)**2-(YY-centerW)**2)/(2*scale**2) )
    return  ZZ

###################################################################################################

use_gpu = True
case = 2
C = 1
dep_U = 4
# # clip bound
log_max = math.log(1e4)
log_min = math.log(1e-8)
import math
if __name__ == '__main__':

    args = parse_args()
    torch.set_default_dtype(torch.float64)
    # load the pretrained model
    print('Loading the Model')
    # checkpoint = torch.load('./model_state/model_state_seismic_v0_3') #c2 150 8.00
    # checkpoint = torch.load('./models_denoise/VDN_40-40/model_state_50') #c2 150 8.19
    # checkpoint = torch.load('./models_denoise/VDN_128-128/model_state_20')# Good  c2 150 9.11
      # depth=5
    # checkpoint = torch.load('./models_denoise/VDN_D5_32/model_state_50') #c2 150 8.28
      #VDN_NestedUNet
    # checkpoint = torch.load('./model_state/model_state_seismic_v1_518') #c2 150 9.33
    # checkpoint = torch.load('./models_denoise/VDN_NestedUNet_32-32/model_state_50')#c2 150 9.95
    # checkpoint = torch.load('./models_denoise/VDN_NestedUNet_128-128/model_state_50')  # c2 150 9.95

    # checkpoint = torch.load('./models_denoise/fielddata/VDN30p/model_state_1')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_35/model_state_18')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/2VUn/model_state_8')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/model_state_18')
    # checkpoint = torch.load('./models_denoise/1111/vdn_unet_75_100000/model_state_34')
    #合成实验图例
    checkpoint = torch.load('./models_denoise/1111/fre/f_35/15syVUn/model_state_26')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_20/13syVUn/model_state_40')
    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/2VUn/model_state_10')  # 6最好

    # checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/2VUn4/model_state_20')#6最好



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
    snrs = []


    #########################################
    im = 'overthrust_204'
    data_dir = 'E:\dataset\\from zhangwei\overthrust\\'
    original = io.loadmat(data_dir + 'overthrust_204.mat')['data']#.astype(np.float32) # 187*801
    # data_dir='E:\博士期间资料\王志强\Marmousi\\'
    # original = io.loadmat(data_dir + 'Marmousi.mat')['data']  # 2441*13601
    # x_max = max(abs(original.max()), abs(original.min()))
    x=original[0:160, 320:448]#[0:160, 320:480] #[1000:1640,7000:7640] #[0:160, 320:480]
    # x = x / x_max

    H, W = x.shape
    if H % 2 ** dep_U != 0:
        H -= H % 2 ** dep_U
    if W % 2 ** dep_U != 0:
        W -= W % 2 ** dep_U
    x = x[:H, :W, ]

    # Generate the sigma map
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
    else:
        sys.exit('Please input the corrected test case: 1, 2 or 3')
    sigma = 10/ 255.0 + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * ((255*0.8-10) / 255.0)
    sigma = cv2.resize(sigma, (W, H))
    # sigma_map = cv2.resize(generate_sigma(), (W, H))
    # # ###########
    np.random.seed(seed=0)  # for reproducibility
    # x = loadmat('./test_data/seismic/clear.mat')['output'][:, :][901:1413, 7001:7513]#[901:1157, 7001:7257]
    # sigma = cv2.resize(sigma, x.shape)
    # x=x/x.max()
    # # #######################
    x_max = max(abs(x.max()), abs(x.min()))
    y = x + np.random.normal(0, 1, x.shape)* sigma[:, :]#*x_max
    # y = x + np.random.normal(0, 1, x.shape) * sigma_map
    y = x + np.random.normal(0, 75 / 255.0, x.shape)*x_max

    # io.savemat(('./noise/noise_case3.mat'), {'data': y[:, :, np.newaxis]})
    # y = loadmat('./noise/noise_case3.mat')['data'].squeeze()
    # ################################
    snr_y = compare_SNR(x, y)
    print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
    psnr_y = compare_psnr(x, y)
    print('psnr_y_before=', '{:.4f}'.format(psnr_y))
    x1 = img_as_ubyte(x.clip(-1, 1))
    y1 = img_as_ubyte(y.clip(-1, 1))
    y_ssim = compare_ssim(y1, x1, data_range=255, gaussian_weights=True,
                          use_sample_covariance=False, multichannel=False)
    print('ssim_before=', '{:.4f}'.format(y_ssim))
    ##################################
    # y=loadmat('./test_data/seismic/pao1.mat')['d'][:, :].clip(-50, 50)[1000:1128,50:178 ]
    # y=y/y.max()
    ################################
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
    torch.cuda.synchronize()
    start_time = time.time()
    if use_gpu:
        y_ = y_.cuda()
        print('Begin Testing on GPU')
    else:
        print('Begin Testing on CPU')
    with torch.autograd.set_grad_enabled(False):
        phi_Z = model(y_, 'test')
        err = phi_Z.cpu().numpy()
        phi_sigma = model(y_, 'sigma')
        phi_sigma.clamp_(min=log_min, max=log_max)
        phi_sigma = phi_sigma  # /phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)
        sigma2 = beta / (alpha + 1)
        sigma2 = sigma2.cpu().numpy()
        # io.savemat(('./noise/GaussSigma.mat'), {'data': np.squeeze(sigma2)})
        sigma = np.sqrt(sigma2).squeeze()
        from datasets.data_tools import sigma_estimate

        print("sigma2.min:", sigma2.min(), "sigma2.median:", np.median(sigma), "sigma2.ave:", np.average(sigma),
              "sigma2.max:", sigma.max())
        m2 = np.exp(err[:, C:, ])  # variance
        m = np.sqrt(m2.squeeze())
        print("m.min:", m.min(), "m.median:", np.median(m), "m.max:", m.max())
    if use_gpu:
        y_ = y_.cpu().numpy()
    else:
        y_ = y_.numpy()
    x_ = y_ - err[:, :C, ]
    x_ = x_.squeeze()
    sigma2 = sigma2.squeeze()
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))
    # #########
    # im_noisy_img = img_as_ubyte(NormMinandMax(x))
    # im_gt_img = img_as_ubyte(NormMinandMax(x_))
    # SSIM=compare_ssim(x, x_, data_range=255, multichannel=False)
    # print(SSIM)
    # ####################
    snr_x_ = compare_SNR(x, x_)
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_x_after=', '{:.4f}'.format(psnr_x_))
    ####################################################
    from skimage.measure import compare_psnr, compare_ssim
    from skimage import img_as_float, img_as_ubyte

    x1 = img_as_ubyte(x.clip(-1, 1))
    x_1 = img_as_ubyte(x_.clip(-1, 1))
    ssim = compare_ssim(x1, x_1, data_range=255, gaussian_weights=True,
                        use_sample_covariance=False, multichannel=False)
    print('ssim=', '{:.4f}'.format(ssim))
    ################################################
    # y_max = max(abs(y.copy().max()), abs(y.copy().min()))
    # x_max=y_max
    if args.save_result:
        name, ext = os.path.splitext(im)
        show(x, y, x_)
        # showsigma(sigma)
        # save_result(x_, path=os.path.join(args.result_dir, name + '_dncnn' + '.png'))  # save the denoised image
        from datasets import wigb
        # wigb.wigb(x.copy())
        # wigb.wigb(y.copy() )
        # wigb.wigb(x_.copy())
        # # wigb.wigb(y.copy() - x.copy())
        # wigb.wigb(y.copy()-x_.copy())
        # # wigb.wigb(x-x_)

        # x_max=1
        # 设置横纵坐标的名称以及对应字体格式
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 30,
                 }

        wigb.wigb((x.copy()) /x_max, figsize=(30, 20),no_plot=True)
        plt.xticks([])
        plt.yticks([])

        # bwith=2
        # ax = plt.gca()  # 获取边框
        # ax.spines['bottom'].set_linewidth(bwith)
        # ax.spines['left'].set_linewidth(bwith)
        # ax.spines['top'].set_linewidth(bwith)
        # ax.spines['right'].set_linewidth(bwith)
        #
        # plt.gca().xaxis.set_ticks(np.arange(0, 129, 30))
        # plt.gca().yaxis.set_ticks(np.arange(0, 161, 40))
        # plt.xlabel('Trace',font2)
        # plt.ylabel('Time sampling number',font2)
        # plt.gca().set_xticklabels(np.arange(320, 449, 30))
        # # 设置坐标刻度值的大小以及刻度值的字体
        # plt.tick_params(labelsize=26)
        # labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]
        # plt.xlim(320, 480)  # 0:160, 320:480
        # plt.ylim(160, 0)
        # 将文件保存至文件中并且画出图
        # plt.savefig('figure.eps')
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\clean.png', format='png', dpi=50,bbox_inches='tight')

        wigb.wigb(y.copy() / x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\c2-noisy.png',
        #             format='png',
        #             dpi=50, bbox_inches='tight')

        wigb.wigb(x_.copy()/ x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\vun-dn-g.png', format='png',
        #             dpi=50, bbox_inches='tight')
        wigb.wigb((y.copy() - x.copy()) / x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\c2-noise.png', format='png',
        #             dpi=50, bbox_inches='tight')
        wigb.wigb((y.copy() - x_.copy()) / x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\vun-n-g.png', format='png',
        #             dpi=50, bbox_inches='tight')
        wigb.wigb((x.copy() - x_.copy()) / x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\vun-r-g.png', format='png',
        #             dpi=50, bbox_inches='tight')









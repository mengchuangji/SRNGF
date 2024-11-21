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

    plt.imshow(sigma2, vmin=0,vmax=1)
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
        scale = pch_size/4
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        # up = random.uniform(10/255.0, 75/255.0)
        # down = random.uniform(10/255.0, 75/255.0)
        up=255/255.0
        down =10/255.0
        if up < down:
            up, down = down, up
        # up += 5/255.0
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

    # checkpoint = torch.load('./models_denoise/fielddata/VDN20p/model_state_1')
    # checkpoint = torch.load('./models_denoise/1111/fre/f7/model_state_20')
    checkpoint = torch.load('./models_denoise/1111/fre/f_s_7/model_state_8')


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

    for im in os.listdir(args.data_dir):
        if im.endswith(".segy") or im.endswith(".sgy"):
            filename = os.path.join(args.data_dir, im)
            with segyio.open(filename,'r',ignore_geometry=True) as f:
                f.mmap()

                sourceX = f.attributes(segyio.TraceField.SourceX)[:]
                trace_num = len(sourceX)#number of trace, The sourceX under the same shot is the same character.
                if trace_num>500:
                    data = np.asarray([np.copy(x) for x in f.trace[0:500]]).T

                    if data.shape[0]>600:
                        x = data[400:600,100:300]
                    else:
                        x = data[:,:]

                else:
                    data = np.asarray([np.copy(x) for x in f.trace[:]]).T

                    if data.shape[0]>600:
                        x = data[400:600,100:300]
                    else:
                        x = data[:,:]
                
                f.close()
#########################################

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
                sigma = sincos_kernel().T
            elif case == 3:
                # Test case 3
                sigma = generate_gauss_kernel_mix(256, 256)
            else:
                sys.exit('Please input the corrected test case: 1, 2 or 3')
            sigma = 10 / 255.0 + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * ((100 - 10) / 255.0)
            sigma = cv2.resize(sigma, (W, H))


            sigma_map = cv2.resize(generate_sigma(), (W, H))
            # io.savemat(('./noise/guasskernel255.mat'), {'data': np.squeeze(sigma_map)})
            # # ###########
            np.random.seed(seed=0)  # for reproducibility
            # x = loadmat('./test_data/seismic/clear.mat')['output'][:, :][901:1413, 7001:7513]#[901:1157, 7001:7257]
            # sigma = cv2.resize(sigma, x.shape)
            # x=x/x.max()
            # # #######################
            # y = x + np.random.normal(0, 1, x.shape)* sigma[:, :]
            # y = x + np.random.normal(0, 1, x.shape) * sigma_map
            y = x + np.random.normal(0, 50 / 255.0, x.shape)

            # io.savemat(('./noise/noise_case3.mat'), {'data': y[:, :, np.newaxis]})
            # y = loadmat('./noise/noise_case3.mat')['data'].squeeze()
            # ################################
            snr_y = compare_SNR(x, y)
            print(' snr_y= {1:2.2f}dB'.format('test',snr_y))
            psnr_y = compare_psnr(x, y)
            print('psnr_y=', '{:.4f}'.format(psnr_y))
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
                phi_sigma = phi_sigma#/phi_sigma.max()
                log_alpha = phi_sigma[:, :C, ]
                alpha = torch.exp(log_alpha)
                log_beta = phi_sigma[:, C:, ]
                beta = torch.exp(log_beta)
                sigma2 = beta / (alpha + 1)
                sigma2 = sigma2.cpu().numpy()
                # io.savemat(('./noise/GaussSigma.mat'), {'data': np.squeeze(sigma2)})
                sigma = np.sqrt(sigma2)
                from  datasets.data_tools import  sigma_estimate

                print("sigma2.min:", sigma2.min(), "sigma2.median:", np.median(sigma),"sigma2.ave:", np.average(sigma), "sigma2.max:", sigma.max())
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
            print('PSNR=', '{:.4f}'.format(psnr_x_))

            ####################################################
            from skimage.measure import compare_psnr, compare_ssim
            from skimage import img_as_float, img_as_ubyte

            x1 = img_as_ubyte(x.clip(-1, 1))
            x_1 = img_as_ubyte(x_.clip(-1, 1))
            ssim = compare_ssim(x1, x_1, data_range=255, gaussian_weights=True,
                                use_sample_covariance=False, multichannel=False)
            print('ssim=', '{:.4f}'.format(ssim))
            ################################################


            if args.save_result:
                name, ext = os.path.splitext(im)
                show(x,y,x_)
                showsigma(sigma2)
                save_result(x_, path=os.path.join(args.result_dir,name+'_dncnn'+'.png'))  # save the denoised image
            snrs.append(snr_x_)
    snr_avg = np.mean(snrs)
    snrs.append(snr_avg)
    if args.save_result:
        save_result(snrs, path=os.path.join(args.result_dir,'results.txt'))
    log('Datset: {0:10s} \n  SNR = {1:2.2f}dB'.format('test', snr_avg))









# -*- coding: utf-8 -*-

import argparse
import random
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from scipy.io import loadmat
from skimage.io import imread, imsave
from seis_util.get_patch import *
import segyio
from seis_util.gain import *
from seis_util.utils import peaks, sincos_kernel, generate_gauss_kernel_mix
import scipy.io as io
from skimage.measure import compare_psnr
from skimage import img_as_ubyte
from skimage.measure import compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim
import  scipy.io as sio
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/test', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=50, type=float, help='noise level')
    parser.add_argument('--agc', default=False, type=bool, help='Agc operation of the data,True or False')
    # parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_sigma100'), help='directory of the model')
    # parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_sigma0_75'),help='directory of the model')
    # parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
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


def show(x, y, x_):
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


class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 2
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

use_gpu = True
case = 2
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

    if use_gpu:
        from networks.UNet import UNet
        model = UNet(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)
        model = torch.nn.DataParallel(model).cuda()
    else:
        from networks.UNet import UNet
        model = UNet(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)




    # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))

    # model = torch.load('./models_denoise/1111/fre/f_35/14syUnMSE/model_050.pth')
    # model = torch.load('./models_denoise/1111/fre/f_35/14syUnMSE-2/model_011.pth')
    # model = torch.load('./models_denoise/1111/fre/f_20/12syUnMSE-2/model_050.pth')
    # model = torch.load('./models_denoise/1111/fre/f_s_7/6UnMSE/model_050.pth')#论文中放的

    model = torch.load('../../model_zoo/MSE-Unet/model_011.pth')



    log('load trained model')

    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)


    #########################################
    im = 'salt_35'
    data_dir = 'E:\VIRI\paper\\1stPaperSE\\revise\output\\'
    newSegyData = data_dir + 'salt_35.sgy'
    f = segyio.open(newSegyData,
                    ignore_geometry=True)
    f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
    original = np.asarray([np.copy(x) for x in f.trace[:]]).T

    x = original[0:160, 0:640]  # 170*649  0:160, 320:448
    x_max = max(abs(original.max()), abs(original.min()))
    x = x / x_max

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
    sigma = 10 / 255.0 + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * ((400- 10) / 255.0)
    sigma = cv2.resize(sigma, (W, H)).astype(np.float32)
    # sigma_map = cv2.resize(generate_sigma(), (W, H))
    # # ###########
    np.random.seed(seed=0)  # for reproducibility
    # x = loadmat('./test_data/seismic/clear.mat')['output'][:, :][901:1413, 7001:7513]#[901:1157, 7001:7257]
    # sigma = cv2.resize(sigma, x.shape)
    # x=x/x.max()
    # # #######################

    x_max = max(abs(x.max()), abs(x.min()))
    y = x + np.random.normal(0, 1, x.shape).astype(np.float32) * sigma[:, :] #* x_max
    # y = x + np.random.normal(0, 1, x.shape) * sigma_map
    # y = x + np.random.normal(0, 30/ 255.0, x.shape).astype(np.float32)#*x_max
    # ######################
    # from npy2segy import npy2segy
    # file_name='salt_35_N'
    # npy2segy(tar_segy='E:\VIRI\paper\\1stPaperSE\\revise\output\\' + file_name + '.sgy',
    #          source_npy=y,
    #          source_segy='E:\dataset\seismic\Model94_shots.segy')
    # file_name = 'salt_35_Y'
    # npy2segy(tar_segy='E:\VIRI\paper\\1stPaperSE\\revise\output\\' + file_name + '.sgy',
    #          source_npy=x,
    #          source_segy='E:\dataset\seismic\Model94_shots.segy')
    # ######################
    y_max=abs(y).max()
    y_max=1
    y = y/y_max
    x = x/y_max

    # io.savemat(('./noise/noise_case3.mat'), {'data': y[:, :, np.newaxis]})
    # y = loadmat('./noise/noise_case3.mat')['data'].squeeze()
    # ################################
    snr_y = compare_SNR(x, y)
    print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
    psnr_y = compare_psnr(x, y)
    print('psnr_y_before=', '{:.4f}'.format(psnr_y))
    # x1 = img_as_ubyte(x.clip(-1, 1))
    # y1 = img_as_ubyte(y.clip(-1, 1))
    y_ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(y_ssim))
    ##################################
    # y=loadmat('./test_data/seismic/pao1.mat')['d'][:, :].clip(-50, 50)[1000:1128,50:178 ]
    # y=y/y.max()
    ################################
    # y.astype(np.float32)
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
    torch.cuda.synchronize()
    start_time = time.time()
    if use_gpu:
        y_ = y_.cuda()
        print('Begin Testing on GPU')
    else:
        print('Begin Testing on CPU')
    x_ = model(y_)  # inferences
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))
    x_ = x_.view(y.shape[0], y.shape[1])
    x_ = x_.cpu()
    x_ = x_.detach().numpy().astype(np.float32)
    torch.cuda.synchronize()

    # ####################
    snr_x_ = compare_SNR(x, x_)
    print(' snr_y_after= {1:2.2f}dB'.format('test', snr_x_))
    snr_x_ = compare_SNR(x, x_)
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_x_after=', '{:.4f}'.format(psnr_x_))
    ####################################################
    # from skimage.measure import compare_psnr, compare_ssim
    # from skimage import img_as_float, img_as_ubyte
    #
    # x1 = img_as_ubyte(x.clip(-1, 1))
    # x_1 = img_as_ubyte(x_.clip(-1, 1))
    ssim = compare_ssim(x, x_)
    print('ssim=', '{:.4f}'.format(ssim))
    ################################################

    if args.save_result:
        name, ext = os.path.splitext(im)
        # show(x, y, x_)

        # sio.savemat(('../../output/results/salt_sc200_' + 'MSE(unet-ng75)' + '_dn.mat'),
        #             {'data': x_[:, :]})

        from seis_util.plotfunction import show_DnNR_3x1, show_DnNR_1x3
        # show_DnNR_3x1(x=data_test,y=noisy_data,x_=denoise_data)
        show_DnNR_1x3(x, y, x_,method='un')

        # showsigma(sigma)
        # save_result(x_, path=os.path.join(args.result_dir, name + '_dncnn' + '.png'))  # save the denoised image
        # from datasets import wigb
        #
        # # wigb.wigb(x.copy()/x.copy().max(),figsize=(30, 15))
        # # wigb.wigb(y.copy()/x.copy().max(),figsize=(30, 15))
        # # wigb.wigb(x_.copy()/x.copy().max(),figsize=(30, 15))
        # # wigb.wigb((y.copy() - x.copy())/(y.copy() - x.copy()).max(),figsize=(30, 15))
        # # wigb.wigb((y.copy() - x_.copy())/(y.copy() - x_.copy()).max(),figsize=(30, 15))
        # # wigb.wigb((x.copy()-x_.copy())/(x.copy()-x_.copy()).max(),figsize=(30, 15))
        #
        # # 设置横纵坐标的名称以及对应字体格式
        # font2 = {'family': 'Times New Roman',
        #          'weight': 'normal',
        #          'size': 30,
        #          }
        #
        # # wigb.wigb((x.copy()) / x_max, figsize=(30, 20), no_plot=True)
        # # plt.xticks([])
        # # plt.yticks([])
        #
        # # bwith=2
        # # ax = plt.gca()  # 获取边框
        # # ax.spines['bottom'].set_linewidth(bwith)
        # # ax.spines['left'].set_linewidth(bwith)
        # # ax.spines['top'].set_linewidth(bwith)
        # # ax.spines['right'].set_linewidth(bwith)
        # #
        # # plt.gca().xaxis.set_ticks(np.arange(0, 129, 30))
        # # plt.gca().yaxis.set_ticks(np.arange(0, 161, 40))
        # # plt.xlabel('Trace',font2)
        # # plt.ylabel('Time sampling number',font2)
        # # plt.gca().set_xticklabels(np.arange(320, 449, 30))
        # # # 设置坐标刻度值的大小以及刻度值的字体
        # # plt.tick_params(labelsize=26)
        # # labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        # # [label.set_fontname('Times New Roman') for label in labels]
        # # plt.xlim(320, 480)  # 0:160, 320:480
        # # plt.ylim(160, 0)
        # # 将文件保存至文件中并且画出图
        # # plt.savefig('figure.eps')
        # # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\clean.png', format='png', dpi=50,bbox_inches='tight')
        #
        # wigb.wigb(y.copy() / x_max, figsize=(30, 20),no_plot=False)
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\\g-noisy.png',
        # #             format='png',
        # #             dpi=50, bbox_inches='tight')
        #
        # wigb.wigb(x_.copy() / x_max, figsize=(30, 20), no_plot= False)
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\un-dn-c2.png', format='png',
        # #             dpi=50, bbox_inches='tight')
        # # wigb.wigb((y.copy() - x.copy()) / x_max, figsize=(30, 20),no_plot=True)
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\\g-noise.png', format='png',
        # #             dpi=50, bbox_inches='tight')
        # wigb.wigb((y.copy() - x_.copy()) / x_max, figsize=(30, 20), no_plot=False)
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\un-n-c2.png', format='png',
        # #             dpi=50, bbox_inches='tight')
        # wigb.wigb((x.copy() - x_.copy()) / x_max, figsize=(30, 20), no_plot=False)
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\un-r-c2.png', format='png',
        # #             dpi=50, bbox_inches='tight')
        #





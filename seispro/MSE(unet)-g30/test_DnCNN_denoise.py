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
from get_patch import *
import segyio
from gain import *
from utils import peaks, sincos_kernel, generate_gauss_kernel_mix
import scipy.io as io
from skimage.measure import compare_psnr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/test', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=50, type=float, help='noise level')
    parser.add_argument('--agc', default=False, type=bool, help='Agc operation of the data,True or False')
    # parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_sigma100'), help='directory of the model')
    parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_sigma0_75'),help='directory of the model')
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
    # io.savemat(('./noise/clean.mat'), {'data': x[:, :, np.newaxis]})

    plt.subplot(162)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    plt.title('noised')
    # plt.colorbar(shrink= 0.5)
    io.savemat(('./noise/noisy_guass_100.mat'), {'data': y[:, :, np.newaxis]})

    plt.subplot(163)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_,vmin=-1,vmax=1,cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)
    io.savemat(('./noise/dncnn_dn_c3_100.mat'), {'data': x_[:, :, np.newaxis]})

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
    io.savemat(('./noise/dncnn_res_c3_100.mat'), {'data': (x_-x)[:, :, np.newaxis]})


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

case = 2
if __name__ == '__main__':

    args = parse_args()
    torch.set_default_dtype(torch.float64)


    # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    model = torch.load(os.path.join(args.model_dir, args.model_name))
    log('load trained model')

    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

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

#                    if args.agc:
#                        data = gain(data,0.004,'agc',0.05,1)
#                    else:
#                        data = data/data.max()
                    if data.shape[0]>600:
                        x = data[400:600,100:300]
                    else:
                        x = data[:,:]

                else:
                    data = np.asarray([np.copy(x) for x in f.trace[:]]).T
#                    if args.agc:
#                        data = gain(data,0.004,'agc',0.05,1)
#                    else:
#                        data = data/data.max()

                    if data.shape[0]>600:
                        x = data[400:600,100:300]
                    else:
                        x = data[:,:]

                f.close()

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
            else:
                sys.exit('Please input the corrected test case: 1, 2 or 3')
            sigma = 10 / 255.0 + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * ((75 - 10) / 255.0)
            sigma = cv2.resize(sigma, x.shape)
            # # ###########
            # np.random.seed(seed=0)  # for reproducibility
            # x = loadmat('./test_data/seismic/clear.mat')['output'][:, :][901:1157, 7001:7257]
            # sigma = cv2.resize(sigma, x.shape)
            # x = x / x.max()
            # # #######################

            ###########
            np.random.seed(seed=0)  # for reproducibility
            # y = x + np.random.normal(0, 1, x.shape) * sigma[:, :]
            y = x + np.random.normal(0, 100/255.0, x.shape)
            # y = loadmat('./noise/noise_case3.mat')['data'].squeeze()
            y_ = torch.from_numpy(y.astype(np.float64)).view(1, -1, y.shape[0], y.shape[1])
            #################################
            snr_y = compare_SNR(x, y)
            print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
            ##################################
            torch.cuda.synchronize()
            start_time = time.time()
            y_ = y_.cuda()
            x_ = model(y_)  # inferences
            x_ = x_.view(y.shape[0], y.shape[1])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float64)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            print(' %10s : %2.4f second' % (im, elapsed_time))

            snr_x_ = compare_SNR(x, x_)
            # io.savemat(('./noise/noise_cese1.mat'), {'data': y})

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
                save_result(x_, path=os.path.join(args.result_dir,name+'_dncnn'+'.png'))  # save the denoised image
            snrs.append(snr_x_)
    snr_avg = np.mean(snrs)
    snrs.append(snr_avg)
    if args.save_result:
        save_result(snrs, path=os.path.join(args.result_dir,'results.txt'))
    log('Datset: {0:10s} \n  SNR = {1:2.2f}dB'.format('test', snr_avg))









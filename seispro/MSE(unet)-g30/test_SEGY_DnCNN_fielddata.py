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
from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_float, img_as_ubyte

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/test/original', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=50, type=float, help='noise level')
    parser.add_argument('--agc', default=False, type=bool, help='Agc operation of the data,True or False')

    # parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_fielddata/DnCNN_real5_test'), help='directory of the model')
    # parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'fielddata/simu/DnCNN'),help='directory of the model')
    parser.add_argument('--model_dir', default=os.path.join('models_denoise', '1111/fre/f_s_7/6UnMSE')) #5DnMSE 6UnMSE


    parser.add_argument('--model_name', default='model_030.pth', type=str, help='the model name')
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

def show(x,y):
    import matplotlib.pyplot as plt
    clip=abs(x).max()
    vmin, vmax = -clip, clip
    plt.figure(figsize=(6,10)) #(12,9)
    plt.subplot(131)
    plt.imshow(x,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic) #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(132)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic)
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(133)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic)
    plt.title('removed noise')
    plt.tight_layout()
    plt.show()

def show_t(x,y):
    import matplotlib.pyplot as plt
    clip=abs(x).max()
    vmin, vmax = -clip, clip
    plt.figure(figsize=(9,10)) #(12,9)
    plt.subplot(141)
    plt.imshow(x,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic) #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(142)
    plt.imshow(x,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic) #plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('original')

    plt.subplot(143)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic)
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(144)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=vmin,vmax=vmax,cmap=plt.cm.seismic)
    plt.title('removed noise')
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

# class DnCNN(nn.Module):
#
#     def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
#         super(DnCNN, self).__init__()
#         kernel_size = 2
#         padding = 1
#         layers = []
#         layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(depth-2):
#             layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
#             layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
#         self.dncnn = nn.Sequential(*layers)
#         self._initialize_weights()
#
#     def forward(self, x):
#         y = x
#         out = self.dncnn(x)
#         return y-out
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.orthogonal_(m.weight)
#                 print('init weight')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)

case = 3
use_gpu = True
if __name__ == '__main__':

    args = parse_args()

    # from networks.residual import DnCNN_Residual
    # model=DnCNN_Residual()



    if use_gpu:
        from networks.UNet import UNet
        model = UNet(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)
        model = torch.nn.DataParallel(model).cuda()
    else:
        from networks.UNet import UNet
        model = UNet(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)


    torch.set_default_dtype(torch.float32)


    # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
    model = torch.load(os.path.join(args.model_dir, args.model_name))
    log('load trained model')
    # model=model.load_state_dict(model['model'])
    model.eval()  # evaluation mode

    # if torch.cuda.is_available():
    #     model = model.cuda()
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir, set_cur)
    snrs = []

    data_dir = 'E:\博士期间资料\田亚军\田博给的数据\\2021.3.27数据'
    data_dir2 = 'E:\博士期间资料\田亚军\田博给的数据\\20220128'
    data_dir3 = 'E:\dataset\seismic'
    im = 'PANKE-INline443'
    # im = '03-MonoNoiAtten-16_DYN_L1901-s11857.sgy'
    # original = readsegy(data_dir, 'PANKE-INline443.sgy')[0:2336,0:768]#[0:1536,0:768] #[2000:2128,19:147]  #2351*787 [100:228,19:147] [1000:1128,19:147] [2000:2128,19:147]
    from seis_util.readsegy import readsegy_ith_agc
    # original = readsegy_ith_agc(data_dir2, 'BGP_00_huangtuyuan_pao.sgy', 19, agc=True)[224:1600,
    #            0:224]  # [448:1020, 0:224]#[0:2992, 0:224]#[640:768, 48:176] [576:822, 16:198]#[448:1020, 0:224] # 3000*224
    original = readsegy(data_dir, '03-MonoNoiAtten-16_DYN_L1901-s11857.sgy')[400:800, 0:224] #3000,224 #[340:468, 48:176]#[640:768, 48:176] [576:822, 16:198] # 2351*787
    y = original.astype(np.float32)

    # original = readsegy_ith_agc(data_dir3, 'shots0001_0200.segy', 1, trace_per_shot=1201, agc=False)[:960,
    #            592:1201]  # 2001*1201
    # original = original / abs(original).max()
    # y = (original + np.random.normal(0, 0.01, original.shape)).astype(np.float32)

    import matplotlib.pyplot as plt
    clip = abs(original).max()
    vmin, vmax = -clip, clip
    plt.figure(figsize=(9, 10))  # (12,9)
    plt.imshow(original, vmin=vmin, vmax=vmax, cmap='gray')  # plt.cm.seismic
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('original')
    plt.show()

    ##################################
    y_max=max(abs(original.max()),abs(original.min()))
    y=y/y_max

    #####################################
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
    torch.cuda.synchronize()
    start_time = time.time()
    if use_gpu:
        y_ = y_.cuda()
        print('Begin Testing on GPU')
    else:
        print('Begin Testing on CPU')
    with torch.autograd.set_grad_enabled(False):
        x_ = model(y_)  # inferences
        x_ = x_.view(y.shape[0], y.shape[1])
        if use_gpu:
            x_ = x_.cpu().numpy()
        else:
            x_ = x_.numpy()
    x_ = x_.astype(np.float32)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))

    no=y_.squeeze().cpu().detach().numpy().astype(np.float32)-x_
    # io.savemat(('./noise/PK-dn-ul-n.mat'), {'data': no[:, :, np.newaxis]})
    # io.savemat(('./noise/PK-dn-ul-dn.mat'), {'data': x_[:, :, np.newaxis]})
    # io.savemat(('./noise/ma_denoise_75_cese3.mat'), {'data': x_})
    ####################################################



    if args.save_result:
        # name, ext = os.path.splitext(im)
        # y=gain(y, 0.004, 'agc', 0.05, 1)
        # x_=gain(x_, 0.004, 'agc', 0.05, 1)
        # show(y,x_)
        # show_t(y, x_)
        # from seis_util.localsimi import localsimi
        # simi = localsimi(y-x_, x_, rect=[5, 5,1], niter=20, eps=0.0, verb=1)#rect=[5, 5, 1], niter=50, eps=0.0, verb=1
        # energy_simi=np.sum(simi**2)/simi.size
        # print("energy_simi=",energy_simi)
        from seis_util.plotfunction import show_xyns,show_yxn_gain
        # show_xyns(y,x_,simi.squeeze())
        # showsigma(sigma2)
        show_yxn_gain(y,x_, agc=False,method='un')










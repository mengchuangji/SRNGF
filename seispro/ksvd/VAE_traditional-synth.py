import cv2
import segyio

from models.unet import UNet

import torch

from PIL import Image
import numpy as np

from utils.image_tool import pil_to_np, np_to_pil, np_to_torch, torch_to_np, torch_to_np_1C, np_to_pil_1C

import bm3d
from skimage.measure import compare_psnr, compare_ssim
from skimage.restoration import estimate_sigma

import matplotlib.pyplot as plt
import math
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def save_hist(x, root):
    x = x.flatten()
    plt.figure()
    n, bins, patches = plt.hist(x, bins=128, density=1)
    plt.savefig(root)
    plt.close()


def save_heatmap(image_np, root):
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img_pil = Image.fromarray((255 * rgb_img).astype(np.uint8))
    rgb_img_pil.save(root)


def sample_z(mean):
    eps = mean.clone().normal_()

    return mean + eps


def save_torch(img_torch, root):
    # save_2D_ndArr(path=os.path.join(result_root + mean_name, '.png'),
    #        np.clip(best_image.squeeze(), -1, 1))
    img_np = torch_to_np(img_torch)
    plt.imsave(root, np.clip(img_np.squeeze(), -1, 1), cmap=plt.cm.seismic)


def save_np(img_np, root):
    plt.imsave(root, np.clip(img_np.squeeze(), -1, 1), cmap=plt.cm.seismic)


def save_compare(clean, noisy, denoise, root, method, epoch):
    # clip = 1  # 显示范围，负值越大越明显
    clip = 1 #abs(noisy).max()
    fontsize = 12
    vmin, vmax = -clip, clip
    # Figure
    # fig = plt.figure(dpi=500, figsize=(26, 3))
    figsize = (20,2)  # 设置图形的大小（12，6）
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize, facecolor='w', edgecolor='k',
                            squeeze=False, sharex=True, dpi=100)
    axs = axs.ravel()  # 将多维数组转换为一维数组
    axs[0].imshow(clean, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    axs[0].set_title('iter:{:04d} Clean'.format(epoch))

    axs[1].imshow(noisy, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    psnr = compare_psnr(clean, noisy)
    ssim = compare_ssim(clean, noisy)
    axs[1].set_title('Noisy\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr, ssim),fontsize=fontsize)

    axs[2].imshow(denoise, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    psnr = compare_psnr(clean, denoise)
    ssim = compare_ssim(clean, denoise)
    axs[2].set_title('Denoised VAE-' + method + '\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr, ssim),fontsize=fontsize)

    noise = noisy - denoise
    axs[3].imshow(noise, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    # noised_psnr = psnr(clean, denoise_data)
    # noised_psnr = round(Denoised_psnr, 2)
    axs[3].set_title('noise VAE' + method,fontsize=fontsize)

    plt.savefig(root + 'compare_epoch_{:04d}.png'.format(epoch), bbox_inches='tight')
    plt.show()


def denoising(noise_im, clean_im, LR=1e-2, sigma=5, rho=1, eta=0.5,
              total_step=20, prob1_iter=500, noise_level=None, result_root=None, f=None):
    input_depth = 1
    latent_dim = 1

    en_net = UNet(input_depth, latent_dim).to(device)
    de_net = UNet(latent_dim, input_depth).to(device)

    parameters = [p for p in en_net.parameters()] + [p for p in de_net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=LR)

    l2_loss = torch.nn.MSELoss().cuda()

    i0 = np_to_torch(noise_im).to(device)
    noise_im_torch = np_to_torch(noise_im).to(device)
    i0_til_torch = np_to_torch(noise_im).to(device)
    Y = torch.zeros_like(noise_im_torch).to(device)

    diff_original_np = noise_im.astype(np.float32) - clean_im.astype(np.float32)
    diff_original_name = 'Original_dis.png'
    save_hist(diff_original_np, result_root + diff_original_name)

    best_psnr = 0

    for i in range(total_step):
        print('第%2d次大循环' % (i + 1))
        ################################# sub-problem 1 ###############################

        for i_1 in range(prob1_iter):
            if i_1 % 10 == 0:
                print('第%4d次小循环' % (i_1))
            optimizer.zero_grad()

            mean = en_net(noise_im_torch)
            # epsilon = torch.randn_like(mean).to(device) #这里不除于255更好
            # out = de_net(mean + epsilon)
            z = sample_z(mean)
            out = de_net(z)

            total_loss = 0.5 * l2_loss(out, noise_im_torch)
            total_loss += 0.5 * (1 / sigma ** 2) * l2_loss(mean, i0)
            total_loss += (rho / 2) * l2_loss(i0 + Y, i0_til_torch)

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                i0 = ((1 / sigma ** 2) * mean.detach() + rho * (i0_til_torch - Y)) / ((1 / sigma ** 2) + rho)

        with torch.no_grad():
            ################################# sub-problem 2 ###############################
            if gaussian_denoiser == "bm3d":
                i0_np = torch_to_np(i0)
                Y_np = torch_to_np(Y)
                sig = noise_level
                i0_til_np = bm3d.bm3d((i0_np + Y_np), sig / 255)
                i0_til_torch = np_to_torch(i0_til_np).to(device)
            elif gaussian_denoiser == "fxdecon":
                import matlab.engine
                eng = matlab.engine.start_matlab()
                # import time
                # start_time = time.time()
                i0_np = torch_to_np(i0)
                Y_np = torch_to_np(Y)
                noisy_data=i0_np + Y_np
                denoise_data = eng.fx_decon(matlab.double(noisy_data.tolist()), matlab.double([0.001]),
                                            matlab.double([15]), matlab.double([0.01]), matlab.double([1]),
                                            matlab.double([100]));
                # elapsed_time = time.time() - start_time
                # print(' %10s : %2.4f second' % ('fedecon', elapsed_time))
                i0_til_np = np.array(denoise_data)
                i0_til_torch = np_to_torch(i0_til_np).to(device)
                print('fxdecon done!')
            elif gaussian_denoiser == "dmssa":
                import matlab.engine
                eng = matlab.engine.start_matlab()
                # import time
                # start_time = time.time()
                i0_np = torch_to_np(i0)
                Y_np = torch_to_np(Y)
                noisy_data=i0_np + Y_np
                denoise_data =eng.fxydmssa(matlab.double(noisy_data.tolist()),0.0,100.0,0.001,64,1.0,0)
                # elapsed_time = time.time() - start_time
                # print(' %10s : %2.4f second' % ('fedecon', elapsed_time))
                i0_til_np = np.array(denoise_data)
                i0_til_torch = np_to_torch(i0_til_np).to(device)
                print('dmssa done!')
            elif gaussian_denoiser == "localorthoDMSSA":
                import matlab.engine
                eng = matlab.engine.start_matlab()
                # import time
                # start_time = time.time()
                i0_np = torch_to_np(i0)
                Y_np = torch_to_np(Y)
                noisy_data = i0_np + Y_np
                denoise_data1 = eng.fxydmssa(matlab.double(noisy_data.tolist()), 0.0, 100.0, 0.001, 64, 1.0, 0);
                noise_data1 = noisy_data - np.array(denoise_data1)
                denoise_data2, noise_data2, low = eng.localortho(matlab.double(denoise_data1),
                                                                 matlab.double(noise_data1.tolist()),
                                                                 matlab.double([[20, 20, 1]]),
                                                                 100,
                                                                 0.0,
                                                                 1, nargout=3)  # 20,0,1
                denoise_data = np.array(denoise_data2)
                # elapsed_time = time.time() - start_time
                # print(' %10s : %2.4f second' % ('fedecon', elapsed_time))
                i0_til_np = np.array(denoise_data)
                i0_til_torch = np_to_torch(i0_til_np).to(device)
                print('localorthoDMSSA done!')
            elif gaussian_denoiser == "curvelet":
                import matlab.engine
                eng = matlab.engine.start_matlab()
                # import time
                # start_time = time.time()
                i0_np = torch_to_np(i0)
                Y_np = torch_to_np(Y)
                noisy_data=i0_np + Y_np
                denoise_data = eng.seismic_curvelet_denoise(matlab.double(noisy_data.tolist()),0.2941);
                # elapsed_time = time.time() - start_time
                # print(' %10s : %2.4f second' % ('fedecon', elapsed_time))
                i0_til_np = np.array(denoise_data)
                i0_til_torch = np_to_torch(i0_til_np).to(device)
                print('curvelet done!')
            elif gaussian_denoiser == "ksvd":
                import matlab.engine
                eng = matlab.engine.start_matlab()
                # import time
                # start_time = time.time()
                i0_np = torch_to_np(i0)
                Y_np = torch_to_np(Y)
                noisy_data=i0_np + Y_np
                param = {'T': 2.0, 'niter': 10.0, 'mode': 1.0, 'K': 64.0}  # 'K':64.0
                denoise_data = eng.yc_ksvd_denoise(matlab.double(noisy_data.tolist()), 1.0, matlab.double([[4, 4, 4]]),
                                                   matlab.double([[2, 2, 2]]), 1.0, param, nargout=1);
                # elapsed_time = time.time() - start_time
                # print(' %10s : %2.4f second' % ('fedecon', elapsed_time))
                i0_til_np = np.array(denoise_data)
                i0_til_torch = np_to_torch(i0_til_np).to(device)
                print('ksvd done!')

            ################################# sub-problem 3 ###############################

            Y = Y + eta * (i0 - i0_til_torch)  # / 255 这里取值多少不影响

            ###############################################################################
            denoise_obj_np = i0_np + Y_np
            noise_section = noise_im_torch - i0_til_torch
            noisy_section = noise_im_torch

            denoise_obj_name = 'denoise_obj_{:04d}'.format(i) + '.png'
            Y_name = 'q_{:04d}'.format(i) + '.png'
            i0_name = 'x_num_epoch_{:04d}'.format(i) + '.png'
            mean_name = 'output_encoder_num_epoch_{:04d}'.format(i) + '.png'
            out_name = 'output_decoder_num_epoch_{:04d}'.format(i) + '.png'
            diff_name = 'after_dis_num_epoch_{:04d}'.format(i) + '.png'
            noise_section_name = 'noise_section_num_epoch_{:04d}'.format(i) + '.png'
            i0_til_name = 'p_num_epoch_{:04d}'.format(i) + '.png'

            save_np(denoise_obj_np, result_root + denoise_obj_name)
            save_torch(noise_section, result_root + noise_section_name)
            save_torch(noisy_section, result_root + 'noisy_section' + '.png')
            save_torch(Y, result_root + Y_name)
            save_torch(mean, result_root + mean_name)
            save_torch(out, result_root + out_name)
            save_torch(i0, result_root + i0_name)
            save_np(i0_til_np, result_root + i0_til_name)

            mean_np = torch_to_np_1C(mean)
            diff_np = mean_np - clean_im
            save_hist(diff_np, result_root + diff_name)

            # i0_til_np = torch_to_np(i0_til_torch).clip(-1, 1)#.clip(0, 255)
            # save_np(i0_til_np,os.path.join(result_root, '{}'.format(i) + '.png'))
            psnr = compare_psnr(clean_im.squeeze(), i0_til_np.squeeze())
            ssim = compare_ssim(clean_im.squeeze(), i0_til_np.squeeze())

            save_compare(clean=clean_im.squeeze(), noisy=noise_im.squeeze(), denoise=torch_to_np(i0_til_torch),
                         root=result_root, method=gaussian_denoiser, epoch=i)
            from seis_util.plotfunction import show_DnNR_1x3
            show_DnNR_1x3(x=clean_im.squeeze(), y=noise_im.squeeze(), x_=torch_to_np(i0_til_torch), method=gaussian_denoiser)

            # i0_til_pil = np_to_pil_1C(np.expand_dims(i0_til_np,0))
            # i0_til_pil.save(os.path.join(result_root, '{}'.format(i) + '.png'))

            print('Iteration: {:02d}, VAE Loss: {:f}, PSNR: {:f}, SSIM: {:f}'.format(i, total_loss.item(), psnr, ssim),
                  file=f, flush=True)

            if best_psnr < psnr:
                best_psnr = psnr
                best_ssim = ssim
            else:
                break

    return i0_til_np, best_psnr, best_ssim


###############################################################################

if __name__ == "__main__":
    LR = 1e-2
    sigma_ = 5
    rho = 1
    eta = 0.5
    total_step = 30
    prob1_iter = 500#500

    psnrs = []
    ssims = []

    # synthesis sample
    gaussian_denoiser = "ksvd"  # nlm fxdecon bm3d curvelet dmssa localorthoDMSSA ksvd
    noisy_name = 'salt-sincos75' #xinjiang

#############################
    path = '..\..\seismic\\test\\'
    clean_name = path +'salt_35.sgy'
    case=2
    # Choose your Gaussian Denoiser mode
    result_root = '..\..\output\\vae_' + gaussian_denoiser + '_' + noisy_name+'\\'
    os.system('mkdir ' + result_root)

    f = segyio.open(clean_name, ignore_geometry=True)
    f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
    original = np.asarray([np.copy(x) for x in f.trace[:]]).T[:160, :640]  # (512,512)
    H, W = original.shape

    # Generate the sigma map
    from seis_util.generateSigmaMap import peaks, gaussian_kernel,sincos_kernel,generate_gauss_kernel_mix,Panke100_228_19_147Sigma,MonoPao
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
    x=original
    x_max = abs(x).max()# 归一化到-1,1之间
    x=x/x_max
    # y = x + np.random.normal(0, 1, x.shape)* sigma[:, :]
    # y = x + np.random.normal(0, 1, x.shape) * sigma_map
    y = x + np.random.normal(0, 30 / 255.0, x.shape)
    # io.savemat(('./noise/noise_case3.mat'), {'data': y[:, :, np.newaxis]})
    # y = loadmat('./noise/noise_case3.mat')['data'].squeeze()
    # ################################
    # y=loadmat('./test_data/seismic/pao1.mat')['d'][:, :].clip(-50, 50)[1000:1128,50:178 ]
    # y=y/y.max()
    ##############################################
    y_max = abs(y).max()
    y = y / y_max
    x = x / y_max
    snr_y = compare_SNR(x, y)
    print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
    psnr_y = compare_psnr(x, y)
    print('psnr_y_before=', '{:.4f}'.format(psnr_y))
    y_ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(y_ssim))
    ##################################
    noisy_im_np = y
    clean_im_np = x
    noise_level = estimate_sigma(y) * 2
    noise_level = 100 #30

    with open(result_root + 'result.txt', 'w') as f:
        _, psnr, ssim = denoising(noisy_im_np, clean_im_np, LR=LR, sigma=sigma_, rho=rho, eta=eta,
                                  total_step=total_step, prob1_iter=prob1_iter,
                                  noise_level=noise_level, result_root=result_root, f=f)

        psnrs.append(psnr)
        ssims.append(ssim)

    with open('..\..\output\\vae_'+gaussian_denoiser+'_'+noisy_name+'\\' + 'psnr_ssim.txt', 'w') as f:
        print('PSNR: {}'.format(sum(psnrs) / len(psnrs)), file=f)
        print('SSIM: {}'.format(sum(ssims) / len(ssims)), file=f)

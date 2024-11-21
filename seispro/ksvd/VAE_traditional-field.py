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

import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    clip = abs(noisy).max()
    fontsize = 12
    vmin, vmax = -clip, clip
    # Figure
    # fig = plt.figure(dpi=500, figsize=(26, 3))
    figsize = (24,5)  # 设置图形的大小（12，6）
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize, facecolor='w', edgecolor='k',
                            squeeze=False, sharex=True, dpi=100)
    axs = axs.ravel()  # 将多维数组转换为一维数组
    axs[0].imshow(clean, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    axs[0].set_title('iter:{:04d} Clean'.format(epoch))

    axs[1].imshow(noisy_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
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
                denoise_data = eng.fx_decon(matlab.double(noisy_data.tolist()), matlab.double([0.004]),
                                            matlab.double([15]), matlab.double([0.01]), matlab.double([1]),
                                            matlab.double([100])); #xinjiang:dt=0.004
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
                denoise_data =eng.fxydmssa(matlab.double(noisy_data.tolist()),0.0,100.0,0.004,64,1.0,0)#filed xinajign dt=0.004
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
                denoise_data1 = eng.fxydmssa(matlab.double(noisy_data.tolist()), 0.0, 100.0, 0.004, 64, 1.0, 0);#dt=0.004
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
            from seis_util.plotfunction import show_DnNR_f_1x3_
            show_DnNR_f_1x3_(x=clean_im.squeeze(), y=noise_im.squeeze(), x_=torch_to_np(i0_til_torch), method=gaussian_denoiser)
             # calcuate localsimilarity
            from seis_util.plotfunction import show_DnNSimi_f_1x3_
            from seis_util.localsimi import localsimi
            simi = localsimi(noise_im.squeeze() - torch_to_np(i0_til_torch), torch_to_np(i0_til_torch), rect=[5, 5, 1],
                             niter=20, eps=0.0, verb=1)
            energy_simi = np.sum(simi ** 2) / simi.size
            print("energy_simi=", energy_simi)
            show_DnNSimi_f_1x3_(x=clean_im.squeeze(), y=noise_im.squeeze(), x_=torch_to_np(i0_til_torch),
                               simi=simi.squeeze(), method='VAE+fxdecon')


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
    # path = './seismic/test/'
    # noises = sorted(glob.glob(path + 'salt_35_N.s*gy'))
    # cleans = sorted(glob.glob(path + 'salt_35_Y.s*gy'))

    # noises = sorted(glob.glob(path + '*test2-X.s*gy'))
    # cleans = sorted(glob.glob(path + '*test2-Y.s*gy'))
    path = '..\..\seismic\\field\\'
    noises = sorted(glob.glob(path + '00-L120-X.s*gy'))
    cleans = sorted(glob.glob(path + '00-L120-Y.s*gy'))

    LR = 1e-2
    sigma = 5
    rho = 1
    eta = 0.5
    total_step = 30
    prob1_iter = 500#500

    psnrs = []
    ssims = []

    gaussian_denoiser = "ksvd"  # nlm fxdecon bm3d curvelet dmssa localorthoDMSSA ksvd
    # noisy_name = 'salt-g30'
    noisy_name = 'xinjiang'

    for noise, clean, in zip(noises, cleans):
        # Choose your Gaussian Denoiser mode
        result_root = '..\..\output\\vae_'+gaussian_denoiser+'_'+noisy_name+'\{}\\'.format(noise.split('\\')[-1][:-4])
        os.system('mkdir ' + result_root)

        f = segyio.open(noise, ignore_geometry=True)
        f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T[492:,0:480]  # (512,512)
        noisy_data_max = abs(data).max()
        noisy_data = data / noisy_data_max  # 归一化到-1,1之间
        noise_im_np = noisy_data.reshape(1, noisy_data.shape[0], noisy_data.shape[1])  # 转换为（1，288，288）

        f = segyio.open(clean, ignore_geometry=True)
        f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T[492:,0:480]  # (512,512)
        clean_data = data / noisy_data_max  # 归一化到-1,1之间
        clean_im_np = clean_data.reshape(1, noisy_data.shape[0], noisy_data.shape[1])  # 转换为（1，288，288）



        # noise_level = estimate_sigma(noise_im_np) * 2
        noise_level = 30
        with open(result_root + 'result.txt', 'w') as f:
            _, psnr, ssim = denoising(noise_im_np, clean_im_np, LR=LR, sigma=sigma, rho=rho, eta=eta,
                                      total_step=total_step, prob1_iter=prob1_iter,
                                      noise_level=noise_level, result_root=result_root, f=f)

            psnrs.append(psnr)
            ssims.append(ssim)

    with open('..\..\output\\vae_'+gaussian_denoiser+'_'+noisy_name+'\\' + 'psnr_ssim.txt', 'w') as f:
        print('PSNR: {}'.format(sum(psnrs) / len(psnrs)), file=f)
        print('SSIM: {}'.format(sum(ssims) / len(ssims)), file=f)

import segyio

from models.unet import UNet

import torch

from PIL import Image
import numpy as np

from utils.image_tool import pil_to_np, np_to_pil, np_to_torch, torch_to_np, torch_to_np_1C, np_to_pil_1C

import bm3d
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim
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


def save_compare(noisy, denoise, root, method, epoch):
    # clip = 1  # 显示范围，负值越大越明显
    clip = abs(noisy).max()
    fontsize = 10
    vmin, vmax = -clip, clip
    # Figure
    # fig = plt.figure(dpi=500, figsize=(26, 3))
    figsize = (10,5)  # 设置图形的大小（12，6）
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, facecolor='w', edgecolor='k',
                            squeeze=False, sharex=True, dpi=300)
    axs = axs.ravel()  # 将多维数组转换为一维数组

    axs[0].imshow(noisy_data, cmap='gray', vmin=vmin, vmax=vmax) #cmap=plt.cm.seismic
    axs[0].set_title('Noisy',fontsize=fontsize)

    axs[1].imshow(denoise, cmap='gray', vmin=vmin, vmax=vmax)
    axs[1].set_title('Denoised GVAE-' + method ,fontsize=fontsize)

    noise = noisy - denoise
    axs[2].imshow(noise, cmap='gray', vmin=vmin, vmax=vmax)
    axs[2].set_title('noise GVAE' + method,fontsize=fontsize)

    plt.savefig(root + 'compare_epoch_{:04d}.png'.format(epoch), bbox_inches='tight')
    plt.show()


def denoising(noise_im, LR=1e-2, sigma=5, rho=1, eta=0.5,
              total_step=20, prob1_iter=500,  result_root=None, f=None):
    input_depth = 1
    latent_dim = 1

    en_net = UNet(input_depth, latent_dim).to(device)
    de_net = UNet(latent_dim, input_depth).to(device)

    if  gaussian_denoiser=='MSE(unet-ng75)':
        # MSE(unet-g75)
        from networks.UNet import UNet as  net
        model = net(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)
        model.load_state_dict(torch.load('./model_zoo/field/MSE-Unet/model_030.pth').module.state_dict(), strict=True)
        model.eval()                #model_011 model_050
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.cuda()
    elif gaussian_denoiser=='MSE(unet-g30)':
        # MSE(unet-ng30)
        from networks.residual import UNet_Residual as net
        model = net(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)
        model=torch.load('./model_zoo/field/MSE-Unet-g30/model_032.pth')
        model = torch.nn.DataParallel(model).cuda()
        model.eval()  #
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.cuda()
    elif gaussian_denoiser=='VI-Non-IID(unet-ng75)':
        # MSE(unet-ng30)
        from networks import VDN
        model = VDN(in_channels=1, dep_U=4, wf=64)
        print('Loading the VI-Non-IID Model')
        checkpoint = torch.load('.\model_zoo\\field\VI-Non-IID-Unet\model_state_10')#f7:10 f35:26
        # checkpoint = torch.load('.\model_zoo\\VI-Non-IID-Unet\\eps1e8sgm75\\model_state_50')  #
        use_gpu=True
        if use_gpu:
            model = torch.nn.DataParallel(model).cuda()
            model.load_state_dict(checkpoint)
        else:
            from utils import load_state_dict_cpu
            load_state_dict_cpu(model, checkpoint)
        model.eval()

    parameters = [p for p in en_net.parameters()] + [p for p in de_net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=LR)

    l2_loss = torch.nn.MSELoss().cuda()

    i0 = np_to_torch(noise_im).to(device)
    noise_im_torch = np_to_torch(noise_im).to(device)
    i0_til_torch = np_to_torch(noise_im).to(device)
    Y = torch.zeros_like(noise_im_torch).to(device)

    # diff_original_np = noise_im.astype(np.float32) - clean_im.astype(np.float32)
    # diff_original_name = 'Original_dis.png'
    # save_hist(diff_original_np, result_root + diff_original_name)

    lowest_loss=0

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
            if gaussian_denoiser == "MSE(unet-g30)":
                i0_til_torch = model(i0 + Y)
                print('unet done!')
            elif gaussian_denoiser == "MSE(unet-ng75)":
                i0_til_torch = model(i0 + Y)
                print('fxdecon done!')
            elif gaussian_denoiser == "VI-Non-IID(unet-ng75)":
                C=1 #channel
                y_=i0 + Y
                phi_Z = model(y_, 'test')
                err = phi_Z
                phi_sigma = model(y_, 'sigma')
                import math
                log_max = math.log(1e4)
                log_min = math.log(1e-8)
                phi_sigma.clamp_(min=log_min, max=log_max)
                phi_sigma = phi_sigma  # /phi_sigma.max()
                log_alpha = phi_sigma[:, :C, ]
                alpha = torch.exp(log_alpha)
                log_beta = phi_sigma[:, C:, ]
                beta = torch.exp(log_beta)
                sigma2 = beta / (alpha + 1)
                sigma2 = sigma2.cpu().numpy()
                # io.savemat(('./noise/GaussSigma.mat'), {'data': np.squeeze(sigma2)})
                sigma_ = np.sqrt(sigma2).squeeze()
                # from datasets.data_tools import sigma_estimate
                print("sigma2.min:", sigma_.min(), "sigma2.median:", np.median(sigma_), "sigma2.ave:", np.average(sigma_),
                      "sigma2.max:", sigma_.max())
                i0_til_torch = y_ - err[:, :C, ]
                print('VI-Non-IID done!')

            ################################# sub-problem 3 ###############################

            Y = Y + eta * (i0 - i0_til_torch)  # / 255 这里取值多少不影响

            ###############################################################################
            denoise_obj_torch = i0 + Y
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

            save_torch(denoise_obj_torch, result_root + denoise_obj_name)
            save_torch(noise_section, result_root + noise_section_name)
            save_torch(noisy_section, result_root + 'noisy_section' + '.png')
            save_torch(Y, result_root + Y_name)
            save_torch(mean, result_root + mean_name)
            save_torch(out, result_root + out_name)
            save_torch(i0, result_root + i0_name)
            save_torch(i0_til_torch, result_root + i0_til_name)

            mean_np = torch_to_np_1C(mean)
            # diff_np = mean_np - clean_im
            # save_hist(diff_np, result_root + diff_name)

            i0_til_np = torch_to_np(i0_til_torch).clip(-1, 1)#.clip(0, 255)
            # save_np(i0_til_np,os.path.join(result_root, '{}'.format(i) + '.png'))

            # save_compare(noisy=noise_im.squeeze(), denoise=torch_to_np(i0_til_torch),
            #              root=result_root, method=gaussian_denoiser, epoch=i)

            from seis_util.plotfunction_noGT import show_DnNR_f_1x3,show_DnNSimi_f_1x3,show_DnNR_f_1x3_,show_DnNSimi_f_1x3_
            # show_DnNR_f_1x3_(y=noise_im.squeeze(), x_=torch_to_np(i0_til_torch), method=gaussian_denoiser)

            from seis_util.plotfunction_noGT_20221116 import show_NyDnNo_single
            # show_NyDnNo_single(noise_im.squeeze(), torch_to_np(i0_til_torch), method=gaussian_denoiser, dpi=300, figsize=(6, 10), clip=0.2)


            # from seis_util.localsimi import localsimi
            # simi = localsimi(noise_im.squeeze() - torch_to_np(i0_til_torch), torch_to_np(i0_til_torch), rect=[5, 5, 1], niter=20, eps=0.0, verb=1)
            # energy_simi = np.sum(simi ** 2) / simi.size
            # print("energy_simi=", energy_simi)
            # show_DnNSimi_f_1x3_(y=noise_im.squeeze(), x_=torch_to_np(i0_til_torch), simi=simi.squeeze(), method='GVAE+VI-Non-IID(Unet)')



            # wigb显示
            from seis_util import wigb
            # from seis_util.gain import gain
            # x__ = gain(noise_im.squeeze().copy(), 0.002, 'agc', 1.2, 2)  # parameter=0.05
            # denoised__ = gain(torch_to_np(i0_til_torch).copy(), 0.002, 'agc', 1.2, 2)
            # noise__ = gain(noise_im.squeeze().copy() - torch_to_np(i0_til_torch).copy(), 0.002, 'agc', 1.2, 2)

            x__ = noise_im.squeeze().copy()[72:400,70:90]  # parameter=0.05 [300:400,0:64]zoom
            denoised__ = torch_to_np(i0_til_torch).copy()[72:400,70:90] #[300:400,0:64]zoom
            noise__ = x__ - denoised__
            x__max=abs(x__).max()
            wigb.wigb(x__/x__max, figsize=(3, 6), linewidth=1) #(18, 30)(30, 18) #(10, 6)zoom
            wigb.wigb(denoised__/x__max, figsize=(3, 6), linewidth=1)
            wigb.wigb(noise__/x__max, figsize=(3, 6), linewidth=1)

            print('Iteration: {:02d}, GVAE Loss: {:f}'.format(i, total_loss.item()),
                  file=f, flush=True)

            # if lowest_loss < total_loss.item():
            #     lowest_loss = total_loss.item()
            # else:
            #     break

    return i0_til_np, lowest_loss

def plot_time_line(shot,time):
    t = np.arange(0, 224, 1)
    plt.rcParams["font.family"] = "Times New Roman"
    fontsize=14
    linewidth=1.2
    fig, ax0 = plt.subplots(nrows=1, figsize=(6, 2))  # ncols = 1
    ax0.plot(t, shot[time, :], 'red', label='Time='+str(time*2+800)+'ms', linewidth=linewidth)
    # ax0.set_xlabel("Trace", fontsize=fontsize)
    # ax0.set_ylabel("Amplitude", fontsize=fontsize)
    # ax0.set_title('Time='+str(time*2+800)+'ms',
    #               fontsize=fontsize)  # trace='+str(trace_num_1)
    ax0.legend(loc='lower right', fontsize=14)
    ax0.set_xticks(np.arange(0, 224, 40))
    ax0.set_xticklabels(np.arange(0, 224, 40), fontdict={'size': fontsize})
    ax0.set_yticks(np.arange(-1, 1.01, 1))
    ax0.set_yticklabels(np.arange(-1, 1.01, 1), fontdict={'size': fontsize})
    axins03 = ax0.inset_axes((0.05, 0.13, 0.28, 0.28))
    # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
    axins03.plot(t[0:64], shot[time, 0:64], 'red', label='clean trace', alpha=0.7)
    axins03.set_xticks(np.arange(0, 64, 30))
    axins03.set_xticklabels(np.arange(0, 64, 30), fontdict={'size': 6})
    axins03.set_yticks(np.arange(-0.05, 0.06, 0.05))
    axins03.set_yticklabels(np.arange(-0.05, 0.06, 0.05), fontdict={'size': 6})
    axins03.grid()
    # 局部显示并且进行连线
    from seis_util.zone_and_linked import zone_and_linked
    zone_and_linked(ax0, axins03, 0, 63, t, [shot[time, :]],
                    'bottom')
    ax0.set_yticks(np.arange(-1, 1.01, 1))
    ax0.set_yticklabels(np.arange(-1, 1.01, 1), fontdict={'size': fontsize})
    fig.tight_layout()
    fig.set_dpi(400)
    # plt.savefig("./trace_compare_DL_60.jpg")
    plt.show()
###############################################################################

if __name__ == "__main__":
    # path = './seismic/test/'
    # noises = sorted(glob.glob(path + 'salt_35_N.s*gy'))
    # cleans = sorted(glob.glob(path + 'salt_35_Y.s*gy'))

    # noises = sorted(glob.glob(path + '*test2-X.s*gy'))
    # cleans = sorted(glob.glob(path + '*test2-Y.s*gy'))
    path = './seismic/field/'
    noises = sorted(glob.glob(path + '03-MonoNoiAtten-16_DYN_L1901-s11857.s*gy'))
    # data_dir2='E:\博士期间资料\田亚军\田博给的数据\\20220128\\'
    # noises = sorted(glob.glob(data_dir2 + 'lyf_cmp100.s*gy'))

    LR = 1e-2
    sigma = 5 # default 5
    rho = 1.0 # default 1
    eta = 0.5 # default 0.5
    total_step = 30
    prob1_iter = 500 #default 500

    lowest_losses = []


    gaussian_denoiser = "VI-Non-IID(unet-ng75)"  # MSE(unet-g30) MSE(unet-ng75) VI-Non-IID(unet-ng75)
    noisy_name = 'huangtuyuan'

    for noise in noises:
        # Choose your Gaussian Denoiser mode
        result_root = '.\output\\GVAE_'+gaussian_denoiser+'_'+noisy_name+'\{}\\'.format(noise.split('\\')[-1][:-4])
        os.system('mkdir ' + result_root)

        f = segyio.open(noise, ignore_geometry=True)
        f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T[400:800, 0:224]#[400:800, 0:224]  #hty:[400:800, 0:224],cmp [0:1200,0:48]   #[492:, 0:480]  # (512,512)
        noisy_data_max = abs(data).max()
        noisy_data = data / noisy_data_max  # 归一化到-1,1之间
        # plot_time_line(noisy_data,390)
        noise_im_np = noisy_data.reshape(1, noisy_data.shape[0], noisy_data.shape[1])  # 转换为（1，288，288）


        # noise_level = estimate_sigma(noise_im_np) * 2

        with open(result_root + 'result.txt', 'w') as f:
            _, lowest_loss = denoising(noise_im_np, LR=LR, sigma=sigma, rho=rho, eta=eta,
                                      total_step=total_step, prob1_iter=prob1_iter,
                                    result_root=result_root, f=f)

            lowest_losses.append(lowest_loss)
    with open('.\output\\GVAE_'+gaussian_denoiser+'_'+noisy_name+'\\' + 'lowest_loss.txt', 'w') as f:
        print('lowest_loss: {}'.format(lowest_loss), file=f)


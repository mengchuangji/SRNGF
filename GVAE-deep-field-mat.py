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
    axs[2].set_title('Denoised GVAE-' + method + '\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr, ssim),fontsize=fontsize)

    noise = noisy - denoise
    axs[3].imshow(noise, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    # noised_psnr = psnr(clean, denoise_data)
    # noised_psnr = round(Denoised_psnr, 2)
    axs[3].set_title('noise GVAE' + method,fontsize=fontsize)

    plt.savefig(root + 'compare_epoch_{:04d}.png'.format(epoch), bbox_inches='tight')
    plt.show()


def denoising(noise_im, clean_im, LR=1e-2, sigma=5, rho=1, eta=0.5,
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

        # plot_3_300(torch_to_np(i0)-clean_im.squeeze())
        plot_3_300(torch_to_np(mean))
        plot_sns(torch_to_np(mean)-clean_im.squeeze()) #z-x
        plot_3_300(torch_to_np(out))
        plot_3_300(torch_to_np(i0))  # x_{i+1}
        plot_3_300(torch_to_np(i0+Y))
        plot_sns(torch_to_np(i0+Y) - clean_im.squeeze())  # y-x

        with torch.no_grad():
            ################################# sub-problem 2 ###############################
            # plot_3_300(torch_to_np(i0+Y))
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
            plot_3_300(torch_to_np(i0_til_torch))
            ################################# sub-problem 3 ###############################
            Y = Y + eta * (i0 - i0_til_torch)  # / 255 这里取值多少不影响
            plot_3_300(torch_to_np(Y))
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
            diff_np = mean_np - clean_im
            save_hist(diff_np, result_root + diff_name)

            i0_til_np = torch_to_np(i0_til_torch).clip(-1, 1)#.clip(0, 255)
            # save_np(i0_til_np,os.path.join(result_root, '{}'.format(i) + '.png'))
            psnr = compare_psnr(clean_im.squeeze(), torch_to_np(i0_til_torch).squeeze())
            ssim = compare_ssim(clean_im.squeeze(), torch_to_np(i0_til_torch).squeeze())

            # save_compare(clean=clean_im.squeeze(), noisy=noise_im.squeeze(), denoise=torch_to_np(i0_til_torch),
            #              root=result_root, method=gaussian_denoiser, epoch=i)
            from seis_util.plotfunction import show_DnNR_f_1x3,show_DnNSimi_f_1x3,show_DnNR_f_1x3_,show_DnNSimi_f_1x3_
            # show_DnNR_f_1x3_(x=clean_im.squeeze(), y=noise_im.squeeze(), x_=torch_to_np(i0_til_torch), method=gaussian_denoiser)
            # # from seis_util.localsimi import localsimi
            # simi = localsimi(noise_im.squeeze() - torch_to_np(i0_til_torch), torch_to_np(i0_til_torch), rect=[5, 5, 1], niter=20, eps=0.0, verb=1)
            # energy_simi = np.sum(simi ** 2) / simi.size
            # print("energy_simi=", energy_simi)
            show_DnNSimi_f_1x3_(x=clean_im.squeeze(), y=noise_im.squeeze(), x_=torch_to_np(i0_til_torch), simi=simi.squeeze(), method='GVAE+VI-Non-IID(Unet)')



            # zn = (torch_to_np(denoise_obj_torch) - torch_to_np(i0_til_torch)).copy().flatten()
            #
            # plt.figure(dpi=300, figsize=(3, 3))
            # from scipy.stats import norm
            # import seaborn as sns
            # # sns.distplot(a=gn, color='green',
            # #              hist_kws={"edgecolor": 'white'})
            # sns.distplot(a=zn, fit=norm, color='blue',
            #              hist_kws={"edgecolor": 'white'})
            # # plt.axis('off')
            # plt.show()
            #
            # plt.figure(dpi=300, figsize=(5, 4))
            # plt.xticks([])  # 去掉横坐标值
            # plt.yticks([])  # 去掉纵坐标值
            # plt.imshow(torch_to_np(denoise_obj_torch), vmin=-1, vmax=1, cmap=plt.cm.seismic)
            # plt.show()

            # # 20221116 wigb显示
            # from seis_util import wigb
            # x__ = noise_im.squeeze().copy()[340:, 300:340]  #[280:, 300:364]  [50:100, 0:64]
            # denoised__ = torch_to_np(i0_til_torch).copy()[340:, 300:340]#[50:100, 0:64]
            # noise__ = x__ - denoised__
            # x__max = abs(x__).max()
            # wigb.wigb(x__ / x__max, figsize=(10, 6), linewidth=1)  # (18, 30)(30, 18)
            # wigb.wigb(denoised__ / x__max, figsize=(10, 6), linewidth=1)
            # wigb.wigb(noise__ / x__max, figsize=(10, 6), linewidth=1)

            # i0_til_pil = np_to_pil_1C(np.expand_dims(i0_til_np,0))
            # i0_til_pil.save(os.path.join(result_root, '{}'.format(i) + '.png'))

            print('Iteration: {:02d}, GVAE Loss: {:f}, PSNR: {:f}, SSIM: {:f}'.format(i, total_loss.item(), psnr, ssim),
                  file=f, flush=True)

            if best_psnr < psnr:
                best_psnr = psnr
                best_ssim = ssim
            # else:
            #     break

    return i0_til_np, best_psnr, best_ssim

def plot_3_300(x):
    plt.figure(dpi=300, figsize=(3, 3))
    plt.imshow(x, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.axis('off')
    plt.show()
def plot_sns(x,color='blue'):
    plt.figure(dpi=300, figsize=(3, 3))
    x = x.copy().flatten()
    from scipy.stats import norm
    import seaborn as sns
    # sns.distplot(a=gn, color='green',
    #              hist_kws={"edgecolor": 'white'})
    sns.distplot(a=x,  color=color,
                 hist_kws={"edgecolor": 'white'}) #fit=norm
    # plt.axis('off')
    plt.show()




###############################################################################

if __name__ == "__main__":
    # path = './seismic/test/'
    # noises = sorted(glob.glob(path + 'salt_35_N.s*gy'))
    # cleans = sorted(glob.glob(path + 'salt_35_Y.s*gy'))

    # noises = sorted(glob.glob(path + '*test2-X.s*gy'))
    # cleans = sorted(glob.glob(path + '*test2-Y.s*gy'))

    path = './seismic/field/'
    noises = sorted(glob.glob(path + 'XJ*noisy.mat'))
    cleans = sorted(glob.glob(path + 'XJ*denoised.mat'))


    LR = 1e-2
    sigma = 5 # default 5
    rho = 1.0 # default 1
    eta = 0.5 # default 0.5
    total_step = 30
    prob1_iter = 1000 #default 500

    psnrs = []
    ssims = []

    gaussian_denoiser = "VI-Non-IID(unet-ng75)"  # MSE(unet-g30) MSE(unet-ng75) VI-Non-IID(unet-ng75)
    noisy_name = 'xinjiang'

    for noise, clean, in zip(noises, cleans):
        # Choose your Gaussian Denoiser mode
        result_root = '.\output\\GVAE_'+gaussian_denoiser+'_'+noisy_name+'\{}\\'.format(noise.split('\\')[-1][:-4])
        os.system('mkdir ' + result_root)

        import scipy.io as sio
        noisy=sio.loadmat(noise)['data'][64:128,64:128] #[0:64,0:64][64:128,0:64][64:128,64:128]
        noisy_data_max = abs(noisy).max()
        noisy_data_max=1
        noisy_data = noisy / noisy_data_max  # 归一化到-1,1之间
        noise_im_np = noisy_data.reshape(1, noisy_data.shape[0], noisy_data.shape[1])  # 转换为（1，288，288）

        clean = sio.loadmat(clean)['data'][64:128,64:128] #[0:64,0:64][64:128,0:64][64:128,64:128]
        clean_data = clean / noisy_data_max  # 归一化到-1,1之间
        clean_im_np = clean_data.reshape(1, noisy_data.shape[0], noisy_data.shape[1])  # 转换为（1，288，288）

        # noise_level = estimate_sigma(noise_im_np) * 2
        plot_3_300(noisy_data)
        plot_3_300(clean_data)
        plot_sns(noisy_data-clean_data,color='red')
        # plot_sns(noisy_data[:16,:16]-clean_data[:16,:16])
        # plot_sns(noisy_data[48:,48:] - clean_data[48:, 48:],color='green')

        with open(result_root + 'result.txt', 'w') as f:
            _, psnr, ssim = denoising(noise_im_np, clean_im_np, LR=LR, sigma=sigma, rho=rho, eta=eta,
                                      total_step=total_step, prob1_iter=prob1_iter,
                                    result_root=result_root, f=f)

            psnrs.append(psnr)
            ssims.append(ssim)

    with open('.\output\\GVAE_'+gaussian_denoiser+'_'+noisy_name+'\\' + 'psnr_ssim.txt', 'w') as f:
        print('PSNR: {}'.format(sum(psnrs) / len(psnrs)), file=f)
        print('SSIM: {}'.format(sum(ssims) / len(ssims)), file=f)

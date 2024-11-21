import cv2
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
    clip = abs(y).max()
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
    axs[2].set_title('Denoised GVAE-' + method + '\n, psnr/ssim={:.2f}/{:.4f}'.format(psnr, ssim),fontsize=fontsize)

    noise = noisy - denoise
    axs[3].imshow(noise, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    # noised_psnr = psnr(clean, denoise_data)
    # noised_psnr = round(Denoised_psnr, 2)
    axs[3].set_title('noise GVAE' + method,fontsize=fontsize)

    plt.savefig(root + 'compare_epoch_{:04d}.png'.format(epoch), bbox_inches='tight')
    plt.show()

def plot_sns(x,color='blue'):
    plt.figure(dpi=300, figsize=(6, 6))
    x = x.copy().flatten()
    from scipy.stats import norm
    import seaborn as sns
    # sns.distplot(a=gn, color='green',
    #              hist_kws={"edgecolor": 'white'})
    sns.distplot(a=x, fit=norm, color=color,
                 hist_kws={"edgecolor": 'white'}) #fit=norm
    # plt.axis('off')
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
        model.load_state_dict(torch.load('./model_zoo/MSE-Unet/model_011.pth').module.state_dict(), strict=True)
        model.eval()                #model_011 model_050
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.cuda()
    elif gaussian_denoiser=='MSE(unet-g30)':
        # MSE(unet-ng30)
        from networks.UNet import UNet as  net
        model = net(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)
        model=torch.load('./model_zoo/MSE-Unet/model_050.pth')
        model = torch.nn.DataParallel(model).cuda()
    elif gaussian_denoiser=='VI-Non-IID(unet-ng75)':
        # MSE(unet-ng30)
        from networks import VDN
        model = VDN(in_channels=1, dep_U=4, wf=64)
        print('Loading the VI-Non-IID Model')
        checkpoint = torch.load('.\model_zoo\VI-Non-IID-Unet\model_state_26')#f7:10 f35:26
        use_gpu=True  #26/ 10
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
        plot_83_300(torch_to_np(mean))
        plot_sns(torch_to_np(mean)-clean_im.squeeze()) #z-x
        plot_83_300(torch_to_np(out))
        plot_83_300(torch_to_np(i0))  # x_{i+1}
        plot_83_300(torch_to_np(i0+Y))
        plot_sns(torch_to_np(i0+Y) - clean_im.squeeze())  # y-x

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
            diff_np = mean_np - clean_im
            save_hist(diff_np, result_root + diff_name)

            i0_til_np = torch_to_np(i0_til_torch).clip(-1, 1)#.clip(0, 255)
            # save_np(i0_til_np,os.path.join(result_root, '{}'.format(i) + '.png'))
            psnr = compare_psnr(clean_im.squeeze(), torch_to_np(i0_til_torch).squeeze())
            ssim = compare_ssim(clean_im.squeeze(), torch_to_np(i0_til_torch).squeeze())

            # save_compare(clean=clean_im.squeeze(), noisy=noise_im.squeeze(), denoise=torch_to_np(i0_til_torch),
            #              root=result_root, method=gaussian_denoiser, epoch=i)
            from seis_util.plotfunction import show_DnNR_1x3
            show_DnNR_1x3(x=clean_im.squeeze(), y=noise_im.squeeze(), x_=torch_to_np(i0_til_torch), method=gaussian_denoiser)
            # from seis_util.localsimi import localsimi
            # simi = localsimi(noise_im.squeeze() - torch_to_np(i0_til_torch), torch_to_np(i0_til_torch), rect=[5, 5, 1],
            #                  niter=20, eps=0.0, verb=1)
            # energy_simi = np.sum(simi ** 2) / simi.size
            # print("energy_simi=", energy_simi)
            # from seis_util import wigb
            # x = clean_im.squeeze()
            # y = noise_im.squeeze()
            # x_ = torch_to_np(i0_til_torch)
            # wigb.wigb(x[:, 80:140].copy())
            # wigb.wigb(y[:, 80:140].copy())
            # wigb.wigb(x_[:, 80:140].copy())
            # # wigb.wigb(y.copy() - x.copy())
            # wigb.wigb(y[:, 80:140].copy() - x_[:, 80:140].copy())
            # # wigb.wigb(x-x_)

            # i0_til_pil = np_to_pil_1C(np.expand_dims(i0_til_np,0))
            # i0_til_pil.save(os.path.join(result_root, '{}'.format(i) + '.png'))


            print('Iteration: {:02d}, GVAE Loss: {:f}, PSNR: {:f}, SSIM: {:f}'.format(i, total_loss.item(), psnr, ssim),
                  file=f, flush=True)

            # if best_psnr < psnr:
            #     best_psnr = psnr
            #     best_ssim = ssim
            # else:
            #     break

    return i0_til_np#, best_psnr, best_ssim
###############################################################################
def plot_83_300(x):
    plt.figure(dpi=300, figsize=(8, 3))
    plt.imshow(x, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    LR = 1e-2
    sigma_ = 5 # default 5
    rho = 1 # default 1
    eta = 1  # default 0.5
    total_step = 30
    prob1_iter = 500 #default 500

    path = './seismic/test/'
    noises = sorted(glob.glob(path + 'salt_35_N.s*gy'))
    cleans = sorted(glob.glob(path + 'salt_35_Y.s*gy'))

    # noises = sorted(glob.glob(path + '*test2-X.s*gy'))
    # cleans = sorted(glob.glob(path + '*test2-Y.s*gy'))



    psnrs = []
    ssims = []

    gaussian_denoiser = "MSE(unet-g30)"  # MSE(unet-g30) MSE(unet-ng75) VI-Non-IID(unet-ng75)
    noisy_name = 'salt-sincos100'

    #############################
    path = '.\seismic\\test\\'
    clean_name = path + 'salt_35.sgy'
    case = 2
    # Choose your Gaussian Denoiser mode
    result_root = '.\output\\GVAE_' + gaussian_denoiser + '_' + noisy_name + '\\'
    os.system('mkdir ' + result_root)

    f = segyio.open(clean_name, ignore_geometry=True)
    f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
    original = np.asarray([np.copy(x) for x in f.trace[:]]).T[:160, :640]  # (512,512)
    H, W = original.shape
    x = original
    x_max = abs(x).max()  # 归一化到-1,1之间
    x=x/abs(x).max()
    # Generate the sigma map
    from seis_util.generateSigmaMap import peaks, gaussian_kernel, sincos_kernel, generate_gauss_kernel_mix, \
        Panke100_228_19_147Sigma, MonoPao

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
    # y = x + np.random.normal(0, 1, x.shape) * sigma[:, :]
    # y = x + np.random.normal(0, 1, x.shape) * sigma_map
    y = x + np.random.normal(0, 30 / 255.0, x.shape)
    # io.savemat(('./noise/noise_case3.mat'), {'data': y[:, :, np.newaxis]})
    # y = loadmat('./noise/noise_case3.mat')['data'].squeeze()
    # ################################

    # plot_83_300(y)
    # plot_83_300(x)
    # plot_sns(y - x, color='red')

    snr_y = compare_SNR(x, y)
    print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
    psnr_y = compare_psnr(x, y)
    print('psnr_y_before=', '{:.4f}'.format(psnr_y))
    y_ssim = compare_ssim(x, y)
    print('ssim_before=', '{:.4f}'.format(y_ssim))
    ##################################
    # y=loadmat('./test_data/seismic/pao1.mat')['d'][:, :].clip(-50, 50)[1000:1128,50:178 ]
    # y=y/y.max()
    ##############################################


    # noise_level = estimate_sigma(noise_im_np) * 2
    noisy_im_np = y
    clean_im_np = x
    with open(result_root + 'result.txt', 'w') as f:
        _, psnr, ssim = denoising(noisy_im_np, clean_im_np, LR=LR, sigma=sigma_, rho=rho, eta=eta,
                                  total_step=total_step, prob1_iter=prob1_iter,
                                  result_root=result_root, f=f)

        psnrs.append(psnr)
        ssims.append(ssim)

    with open('.\output\\GVAE_'+gaussian_denoiser+'_'+noisy_name+'\\' + 'psnr_ssim.txt', 'w') as f:
        print('PSNR: {}'.format(sum(psnrs) / len(psnrs)), file=f)
        print('SSIM: {}'.format(sum(ssims) / len(ssims)), file=f)

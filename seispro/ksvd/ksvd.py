import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import segyio
from psnr import psnr

class KSVD(object):
    def __init__(self, n_components, max_iter=100, tol=1e-6,
                 n_nonzero_coefs=None):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        初始化字典矩阵
        """
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
        KSVD迭代过程
        """
        self._initialize(y)
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)

        self.sparsecode = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode


if __name__ == '__main__':
    #############测试地震数据
    file_list='../../seismic/test/00-L120-Y.sgy' #'../data/test1.sgy'
    f = segyio.open(file_list, ignore_geometry=True)
    f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
    data = np.asarray([np.copy(x) for x in f.trace[:]]).T#(512,512)
    data_test = data/abs(data).max()#归一化到-1,1之间
    
    file_list1='../../seismic/test/00-L120-X.sgy' #'../data/noise.sgy'
    f = segyio.open(file_list1, ignore_geometry=True)
    f.mmap()#mmap将一个文件或者其它对象映射进内存，加快读取速度
    data = np.asarray([np.copy(x) for x in f.trace[:]]).T#(512,512)
    noisy_data = data#(288,288)
    noisy_data = noisy_data/abs(noisy_data).max()#归一化到-1,1之间
    ########降噪处理
    ksvd = KSVD(30)
    dictionary, sparsecode = ksvd.fit(noisy_data)
    denoise_data=dictionary.dot(sparsecode)
    noise_data=noisy_data-denoise_data
     #########################画图地震数据图      
    clip = 1e-0#显示范围，负值越大越明显
    vmin, vmax = -clip, clip
            # Figure
    figsize=(12, 3)#设置图形的大小
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize, facecolor='w', edgecolor='k',
                                   squeeze=False,sharex=True,dpi=100)
    axs = axs.ravel()#将多维数组转换为一维数组
    axs[0].imshow(data_test, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    axs[0].set_title('Clear')
    
    axs[1].imshow(noisy_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    noisy_psnr = psnr(data_test,noisy_data)
    noisy_psnr=round(noisy_psnr, 2)
    axs[1].set_title('Noisy, psnr='+ str(noisy_psnr))
    
    axs[2].imshow(denoise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    Denoised_psnr = psnr(data_test,denoise_data)
    Denoised_psnr=round(Denoised_psnr, 2)
    axs[2].set_title('Denoised ksvd, psnr='+ str(Denoised_psnr))

    axs[3].imshow(noise_data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    Denoised_psnr = psnr(data_test, noise_data)
    Denoised_psnr = round(Denoised_psnr, 2)
    axs[3].set_title('Noise ksvd, psnr=' + str(Denoised_psnr))

    plt.show()

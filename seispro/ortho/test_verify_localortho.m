clc;clear;close all;
%% 3D
load('datas.mat');

d11=fxydmssa(dn,0,120,0.004,3,1,0);
norm(d1(:)-d11(:))

figure; yc_imagesc([dn(:,:),d1(:,:),dn(:,:)-d1(:,:)]);
figure; yc_imagesc([d1(:,:),d11(:,:),d1(:,:)-d11(:,:)],90);

yc_snr(d0,dn,2)
yc_snr(d0,d1,2) %20.8644
yc_snr(d0,d11,2)

yc_snr(d0,d2,2) %24.1185

addpath(genpath('~/MATortho'));
rect=[10,10,10];
eps=0;
niter=20;
verb=1;
[d22,noi22,low]=localortho(d1,noi1,rect,niter,eps,verb);% a little different
yc_snr(d0,d22,2) %23.2585

figure;figure; yc_imagesc([d2(:,:),d22(:,:),d2(:,:)-d22(:,:)],90);



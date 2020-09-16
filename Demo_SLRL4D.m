clc; clear;
addpath('SLRL4D');
addpath('BM4D_mex');
addpath('Dataset');
addpath('Quantitative Assessment');

% Simulated Case 1
load('ORI_WDC.mat');
[nr,nc,L] = size(Img);
Img = Img./repmat(max(max(Img,[],1),[],2),nr,nc);
sigma = 0.25;
Noisy_Img = Img + sigma*randn(nr,nc,L);

X = Img./repmat(max(max(Img,[],1),[],2),nr,nc);
Y = Noisy_Img;
clear Img Noisy_Img;

% Parameter Setting
k_num=4;
lambda=1;
iterations = 100;

[Ys,~,~,mpsnr_i,mssim_i] =SLRL4D_fast(Y,'LAMBDA_S',lambda,'SUBSPACE_DIM',k_num,'AL_ITERS',iterations,'TRUE_X',X);

[mpsnr,psnr] = MPSNR(X,Ys);
[mssim,ssim] = MSSIM(X,Ys);
[mfsim,fsim] = MFSIM(X,Ys);
ergas = ErrRelGlobAdimSyn(X,Ys);
msa = MSA(X, Ys);
disp(['PSNR=' num2str(mpsnr)]);
disp(['SSIM=' num2str(mssim)]);
disp(['FSIM=' num2str(mfsim)]);
disp(['ERGAS=' num2str(ergas)]);
disp(['MSA=' num2str(msa)]);
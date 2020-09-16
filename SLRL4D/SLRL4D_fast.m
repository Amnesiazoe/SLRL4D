%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a fast version of SLRL4D denoiser. Using the code should cite the following papers
%
% Sun, L.; He, C.; Zheng, Y.; Tang, S. SLRL4D: Joint Restoration of Subspace Low-Rank Learning and Non-Local 4-D Transform Filtering for Hyperspectral Image. Remote Sens. 2020, 12, 2979.
% L. Sun and B. Jeon, A novel subspace spatial-spectral low rank learning method for hyperspectral denoising, 2017 IEEE Visual Communications and Image Processing (VCIP), St. Petersburg, FL, 2017, pp. 1-4.
% L. Sun, B. Jeon, B. N. Soomro, Y. Zheng, Z. Wu and L. Xiao, Fast Superpixel Based Subspace Low Rank Learning Method for Hyperspectral Denoising, IEEE Access, vol. 6, pp. 12031-12043, 2018.
%
% @article{lesun2020slrl4d,
%  title={SLRL4D: Joint Restoration of Subspace Low-Rank Learning and Non-Local 4-D Transform Filtering for Hyperspectral Image},
%  author={Sun, Le and He, Chengxun and Zheng, Yuhui and Tang, Songze},
%  journal={Remote Sensing},
%  volume={12},
%  number={18},
%  pages={2979},
%  year={2020},
%  publisher={Multidisciplinary Digital Publishing Institute}
% }
% 
% @inproceedings{sun2017novel,
%   title={A novel subspace spatial-spectral low rank learning method for hyperspectral denoising},
%   author={Sun, Le and Jeon, Byeungwoo},
%   booktitle={2017 IEEE Visual Communications and Image Processing (VCIP)},
%   pages={1--4},
%   year={2017},
%   organization={IEEE}
% }
% 
% @article{sun2018fast,
%   title={Fast superpixel based subspace low rank learning method for hyperspectral denoising},
%   author={Sun, Le and Jeon, Byeungwoo and Soomro, Bushra Naz and Zheng, Yuhui and Wu, Zebin and Xiao, Liang},
%   journal={IEEE Access},
%   volume={6},
%   pages={12031--12043},
%   year={2018},
%   publisher={IEEE}
% }
% 
% Feel free to contact me if there is anything we can help
% cxunhey@nuist.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function   [X,Z,E, PSNR_iter,SSIM_iter] = SLRL4D_fast(Y,varargin)
[M,N,L] = size(Y);
lambda_s = 1/sqrt(max(M*N,L));
AL_iters = 50;
sub_dim = 10;
sigma = 0.0032;
%mu = sigma * (sqrt(M*N)+sqrt(L));
verbose = 'off';
true_x = 0;
mu = (sqrt(M*N)+sqrt(L))*sigma;
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'LAMBDA_S'  
                lambda_s = varargin{i+1};
                if lambda_s < 0
                    error('lambda must be non-negative');
                end
            case 'GAMMA' 
                gamma_s = varargin{i+1};
                if gamma_s < 0
                    error('gamma must be non-negative');
                end
            case 'SUBSPACE_DIM'
                sub_dim = varargin{i+1};
                if sub_dim <= 0
                    error('The dimension of the subspace should be nonnegative');
                end
            case 'AL_ITERS'
                AL_iters = round(varargin{i+1});
                if (AL_iters <= 0 )
                    error('AL_iters must a positive integer');
                end
            case 'TRUE_X'
                XT = varargin{i+1};
                true_x = 1;
            case 'VERBOSE'
                verbose = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end
Y = reshape(Y,M*N,L);
Y2D = reshape(Y,[M*N L])';
noise_type='additive';
verbose1 = 'on';
[w Rn] = estNoise(Y2D,noise_type,verbose1);
[kf, Ek,E_all]=hysime(Y2D,w,Rn,verbose1); 
if ~strcmp(verbose1,'on'),fprintf(1,'The signal subspace dimension is: k = %d\n',kf);end
E = E_all(:,1:sub_dim);
E = E'; 
Z = zeros(M*N,sub_dim);
%E = zeros(sub_dim,L);
%S = zeros(M*N,L);
PSNR_iter = zeros(AL_iters,1);
SSIM_iter=zeros(AL_iters,1);
iter = 1;
while iter<=AL_iters
    % estimate the variable S
    S = MySoftTh(Y - Z*E,lambda_s*mu);
    %estimate the variable Z
    Z_est = (Y-S)*E';
    Z_max = max(Z_est(:));
    Z_min = min(Z_est(:));
    Z_est = (Z_est-Z_min)./(Z_max-Z_min); % scale the data to [0 -255]
    [Z, ~] = bm4d(reshape(Z_est,M,N,sub_dim), 'Gauss', 0);
    Z = reshape(Z,M*N,sub_dim).*(Z_max-Z_min) + Z_min;
    %estimate E
    E_est = (Y-S)'*Z;
    [U,Sig,V] = svd(E_est,'econ');
    E = V*U';
    if strcmp(verbose,'on') && true_x
        X_iter = Z*E;
        X_iter = reshape(X_iter,M,N,L);
        PSNR_iter(iter) = MPSNR(XT,X_iter);
        SSIM_iter(iter) = MSSIM(XT,X_iter);
        res = sqrt(sum((X_iter(:)-XT(:)).^2));
        fprintf(strcat(sprintf('iter = %i - ||X - XT|| = %2.3f, MPSNR = %2.3f, MSSIM = %2.3f ',iter, res,PSNR_iter(iter),SSIM_iter(iter)),'\n'));
    end
    iter =  iter+1;
end
X = Z*E;
X = reshape(X,M,N,L);
end
function X= MySoftTh(B,lambda)
X=sign(B).*max(0,abs(B)-(lambda));
end
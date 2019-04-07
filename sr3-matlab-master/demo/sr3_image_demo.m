%% SR3 Image Deconvolution Demo
%
% In this file we demonstrate a more complicated use of the
% SR3 framework. We have wrapped these steps up in the
% function |sr3_deconvtv|.
%
% Because of the flexibility in choosing $C$, it is possible to
% create a simple-minded image deblurring utility using the $\ell_0$ 
% or $\ell_1$ penalties. Further, acceleration is possible, which
% we will demonstrate.
%
% In this example, we load an image, blur it with a Gaussian convolution,
% corrupt it with noise, define a smoothness penalty and solve using 
% SR3.
%
% $A$ corresponds to blurring (this is done with fft calls) and 
% $C$ corresponds to x and y differences, stacked on top of each other.
% The TV-like norm is then given by applying a vector version of either the 
% l1 or l0 penalties to D*w. The $x$ corresponds to a smoothed out version 
% of the original image. Using the $\ell_0$ penalty, the $x$ is 
% cartoon-like (large flat regions) and the non-zero entries in $w$ 
% correspond to edges. The smoothing with $\ell_1$ is less extreme (total 
% variation denoising a la Rudin, Osher, Fatemi)
% 

% load the popular 'cameraman' image
b = double(imread('cameraman.tiff'));
[m,n] = size(b);

% seed random number generator (consistent runs)
iseed = 8675309;
rng(iseed);

% parameters of blurring kernel
blursigma = 3.0; % controls width of standard deviation (in pixels)
nblur = round(2*blursigma); % gives size of filter (two standard deviations here)

% other parameters
%noise parameters
sigma = 2;


% regularization parameters

lam1 = 0.125;
kap1 = 0.5/sigma;

% optimization parameters
itm = 40;

% set up Gaussian blurring
inds = -nblur:nblur;
blurmat1 = inds.^2 + (inds.^2).';
blurmat1 = exp(-0.5*blurmat1/blursigma^2);
bscale = sum(sum(abs(blurmat1)));
blurmat1 = blurmat1/bscale;
blurmat = zeros(m,n);
blurmat(1:2*nblur+1,1:2*nblur+1) = blurmat1;
blurmat = circshift(blurmat,[-(nblur),-(nblur)]);
blurhat = fftn(blurmat);

% perform blurring (this is like applying A)
bhat = fftn(b);
bblurhat = blurhat.*bhat;
bblur = ifftn(bblurhat);

% rhs is blurred image plus noise
noise = randn(m,n)*sigma;
rhs = bblur+noise;

% least squares solution (no regularization)
no_reg = ifftn(fftn(rhs)./blurhat);

snr_rhs = 20*log10(norm(rhs,'fro')/norm(noise,'fro'));

        % solve with vanilla proximal gradient descent
mode = '1';
[x1,w1,stats1] = sr3_deconvtv(blurmat1,rhs,'itm',itm,'lam',lam1, ...
    'kap',kap1,'modereg',mode,'ptf',10,'ifstdtvobj',0);

            % solve with acceleration
mode = '1';
[x1a,w1a,stats1a] = sr3_deconvtv(blurmat1,rhs,'itm',itm,'lam',lam1, ...
    'kap',kap1,'modereg',mode,'ptf',10,'ifstdtvobj',0,'accelerate',true);

% signal-to-noise ratio of recovered images
snr_sr3 = 20*log10(1/(norm(x1-b,'fro')/norm(b,'fro')))
snr_sr3a = 20*log10(1/(norm(x1a-b,'fro')/norm(b,'fro')))

% sparsity of w (thresholded TV derivative of the image)
nnz(w1)
nnz(w1a)

%% Convergence plot
% the accelerated method makes faster progress toward the solution

figure(1)
hold off
clf

objs_sr31 = stats1.objs;
objs_sr31a = stats1a.objs;
msr = length(objs_sr31);

mshow = min(msr,75);

minall = min([min(objs_sr31),min(objs_sr31a)]);
maxall = max([max(objs_sr31),max(objs_sr31a)]);

hold on
semilogy((objs_sr31(1:mshow)-minall)/(maxall-minall),'^r')
semilogy((objs_sr31a(1:mshow)-minall)/(maxall-minall),'ob')

legend('Vanilla SR3','Accelerated SR3')

%% Recovered images
% the recovered images are visually similar, and are a huge improvement
% over the unregularized least squares solution, which is gibberish

figure(2)
hold off

mstart = 75;
mend = 175;
nstart = 175;
nend = 325;

subplot(2,2,1)
h=pcolor(flipud(b(mstart:mend,nstart:nend)));
set(h,'EdgeColor','none')
colormap('gray')
axis equal
axis tight
caxis([-128 383])
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
subplot(2,2,2)
h=pcolor(flipud(no_reg(mstart:mend,nstart:nend,1)));
set(h,'EdgeColor','none')
colormap('gray')
axis equal
axis tight
caxis([-128 383])
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
subplot(2,2,3)
h=pcolor(flipud(x1(mstart:mend,nstart:nend,1)));
set(h,'EdgeColor','none')
colormap('gray')
axis equal
axis tight
caxis([-128 383])
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
subplot(2,2,4)
h=pcolor(flipud(x1a(mstart:mend,nstart:nend,1)));
set(h,'EdgeColor','none')
colormap('gray')
axis equal
axis tight
caxis([-128 383])
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
function [x,w,stats] = sr3_deconvtv(cmat,img,varargin)
%SR3_DECONVTV SR3-style relaxed pursuit method for total variation 
% regularized deconvolution
% 
% Required input (positional):
%
%   cmat   convolution kernel (double precision matrix), must have 
%          odd-valued dimensions (so that it can be centered)
%   img   double precision real matrix (size M by N) representing
%         an image (only a single color channel is supported for now)
%
% Parameter input:
%
%   'lam'       hyper-parameter, control strength of R (default 1.0)
%   'kap'       hyper-parameter, control strength of the quadratic penalty
%               (default 1.0)
%   'itm'       maximum number of iterations (default 100)
%   'tol'       terminate if change in w (in l2 norm) is less than tol
%               (default 1e-6)
%   'ptf'       print every ptf iterations (don't print if 0). (default 0)
%   'modefit'   '2': (default '2')
%   'modereg'   '2': R = 0.5*squared 2 norm, i.e. 0.5*sum(abs(x).^2)
%               '1': R = 1 norm, i.e. sum(abs(x))
%               '0': R = 0 norm, i.e. nnz(x)
%               'mixed': R = sum of 0, 1, and squared 2 norms with 
%                weights l0w, l1w, and l2w
%               'other': R and Rprox must be provided
%               (default '1')
%   'ifobjhis'  flag, if nonzero, store the history of the values of the 
%               objective in stats.objhis; otherwise, these aren't stored
%   'ifstdtvobj'flag, if nonzero, then compute the objective as if 
%               the standard tv regularized problem were being optimized
%
% output:
%   x, w the computed minimizers of the objective
%   stats - some basic statistics of the optimization
%   stats.iter = number of iterations
%   stats.errhis = difference between a given iteration and the next
%   stats.objhis = objective value at a given iteration
%
% Example:
%
%   >> m = 100; n = 2000; k = 10;
%   >> A = randn(m,n);
%   >> y = zeros(n,1); y(randperm(n,k)) = sign(randn(k,1));
%   >> lam = A.'*b;
%   >> [x,w] = rrlsq(A,b,'lam',lam);
%
% See also LASSO, LINSOLVE

% Copyright 2018 Travis Askham and Peng Zheng
% Available under the terms of the MIT License

%% parse inputs

[m,n] = size(img);
[m1,n1] = size(cmat);

[p,R,Rprox] = sr3_deconvtv_parse_input(img,varargin{:});

lam = p.Results.lam;
kap = p.Results.kap;
itm = p.Results.itm;
tol = p.Results.tol;
ptf = p.Results.ptf;
modefit = p.Results.modefit;
modereg = p.Results.modereg;
ifstdtvobj = p.Results.ifstdtvobj;
x = p.Results.xinit;
w = p.Results.winit;
accelerate = p.Results.accelerate;

%% pre-process operators

alpha = lam/kap;

% convolution operator
cmatbig = zeros(m,n);
cmatbig(1:m1,1:n1) = cmat;
cmatbig = circshift(cmatbig,[-((m1-1)/2),-((n1-1)/2)]);
cmathat = fftn(cmatbig);

% difference operators 
dfx = [-1 1];
dfy = [-1; 1];
dfxhat = fftn(dfx,[m,n]);
dfyhat = fftn(dfy,[m,n]);

% precompute a transpose times b
atb = ifftn(conj(cmathat).*fftn(img));

% system matrix (in frequency space) for least squares solve
atadtdhat = abs(cmathat).^2 + kap*(abs(dfxhat).^2 + abs(dfyhat).^2);

xm = x;
wm = w;
vtop = reshape(w(1:m*n),m,n);
vbot = reshape(w(m*n+1:end),m,n);

noi = 0;

err = 2.0*tol;
errs = zeros(itm,1);
objs = zeros(itm,1);

y = zeros(2*m*n,1);

wm  = w;
vm  = w;
am  = 1.0;

while err >= tol
    % solve least squares problem for x
    temp1 = atb + kap*ifftn(conj(dfxhat).*fftn(vtop) ...
        + conj(dfyhat).*fftn(vbot)); 
    xhat = fftn(temp1)./atadtdhat;
    x = ifftn(xhat);
    %grad = ifftn(conj(cmathat).*(cmathat.*xhat-fftn(img))) ...
    %        + kap*(ifftn(abs(dfxhat).^2.*xhat+abs(dfyhat).^2.*xhat)- ...
    %        ifftn(conj(dfxhat).*fftn(wtop) + conj(dfyhat).*fftn(wbot)));
    %fprintf('|g| %e\n',norm(grad,'fro'))
            
    % compute w as prox of C*x
    temp1 = ifftn( dfxhat.*xhat );
    y(1:m*n) = temp1(:);
    temp1 = ifftn( dfyhat.*xhat );
    y(m*n+1:end) = temp1(:);
    w = Rprox(y,alpha);
%     wtop = reshape(w(1:m*n),m,n);
%     wbot = reshape(w(m*n+1:end),m,n);
    
    % acceleration step
    if accelerate
        a = 0.5*(1 + sqrt(1+4*am^2));
        v = w + (am - 1)/a*(w - wm);
        am = a;
    else
        v = w;
    end
    
    vtop = reshape(v(1:m*n),m,n);
    vbot = reshape(v(m*n+1:end),m,n);
    
    
    
    % print and store convergence information
    noi = noi + 1;
    %err = norm(x-xm,'fro')/norm(xm,'fro');
    err = norm(w-wm,'fro')/norm(wm,'fro');
    errs(noi) = err;
    if ifstdtvobj
        obj = 0.5*norm(ifftn(cmathat.*xhat)-img,'fro')^2 ...
            + lam*R(y);
    else
        obj = 0.5*norm(ifftn(cmathat.*xhat)-img,'fro')^2 ...
            + lam*R(w) + 0.5*kap*sum(abs(y-w).^2);
    end
    
    xm = x;
    wm = w;
    vm = v;
    
    
    objs(noi) = obj;
    if mod(noi, ptf) == 0
        fprintf('iter %4d, obj %1.2e, err %1.2e\n', noi, obj, err);
    end
    
    if noi >= itm
        break;
    end
    
end

stats.noi = noi;
stats.errs = errs(1:noi);
stats.objs = objs(1:noi);

end

function [p,R,Rprox] = sr3_deconvtv_parse_input(img,varargin)
%SR3_DECONVTV_PARSE_INPUT parse the input to SR3_DECONVTV
% Sets default values and checks types (within reason)
% See also SR3_DECONVTV for details

    [m,n] = size(img);

    defaultlam = 1.0;
    defaultkap = 1.0;
    defaultitm = 100;
    defaulttol = 1e-6;
    defaultptf = 0;
    defaultmodefit = '2';
    defaultmodereg = '1';
    defaultifstdtvobj = 1;
    defaultaccelerate = false;
    defaultl0w = 0.0;
    defaultl1w = 0.0;
    defaultl2w = 0.0;
    defaultxinit = img;
    defaultwinit = zeros(2*m*n,1);
    
    p = inputParser;
    isdouble = @(x) isa(x,'double');
    isdoublep = @(x) isa(x,'double') && x > 0;
    isdoublepp = @(x) isa(x,'double') && x >= 0;
    isnumericp = @(x) isnumeric(x) && x > 0;
    isnumericpp = @(x) isnumeric(x) && x >= 0;    
    
    addParameter(p,'lam',defaultlam,isdoublep);
    addParameter(p,'kap',defaultkap,isdoublep);
    addParameter(p,'itm',defaultitm,isnumericp);
    addParameter(p,'tol',defaulttol,isdoublep);
    addParameter(p,'ptf',defaultptf,isnumericpp);
    addParameter(p,'modefit',defaultmodefit,@ischar);
    addParameter(p,'modereg',defaultmodereg,@ischar);
    addParameter(p,'ifstdtvobj',defaultifstdtvobj,@isnumeric);
    addParameter(p,'accelerate',defaultaccelerate,@islogical);
    addParameter(p,'l0w',defaultl0w,isdoublepp);
    addParameter(p,'l1w',defaultl1w,isdoublepp);
    addParameter(p,'l2w',defaultl2w,isdoublepp);
    addParameter(p,'xinit',defaultxinit,isdouble);
    addParameter(p,'winit',defaultwinit,isdouble);    
    
    parse(p,varargin{:});
    
    % override if mode '0' '1' or '2' selected
    if strcmp(p.Results.modereg,'0')
        l0w = 1; l1w = 0; l2w = 0;
    elseif strcmp(p.Results.modereg,'1')
        l0w = 0; l1w = 1; l2w = 0;
    elseif strcmp(p.Results.modereg,'2')
        l0w = 0; l1w = 0; l2w = 1;
    else
        l0w = p.Results.l0w; l1w = p.Results.l1w; l2w = p.Results.l2w;
    end
    
    
    if strcmp(p.Results.modereg,'0') || strcmp(p.Results.modereg,'1') ...
            || strcmp(p.Results.modereg,'2') || strcmp(modereg,'mixed')
        if (abs(l0w) == 0 && abs(l1w) == 0 && abs(l2w) == 0)
            warning(['all weights in mixed norm are zero', ...
                '\n prox operation does nothing'])
        end
        R = @(x) l012vecRprox(x,1,l0w,l1w,l2w,0,2);
        Rprox = @(x,alpha) l012vecRprox(x,alpha,l0w,l1w,l2w,1,2);
    else
        error('incorrect value for mode')
    end

end

function [x, w, noi] = sr3(A,b,varargin)
%SR3 Relaxed pursuit method for regularized least squares problems
% of the form:
%   0.5*norm(A*x-b,2)^2 + lam*R(w) + 0.5*kap*norm(C*x-w,2)^2
% over x and w. The output w represents a regularized solution of 
% the least squares problem described by A and b. 
%
% Required input (positional):
%
%   A   double precision real or complex matrix (dimension, say, MxN)
%   b   double precision real or complex vector (length M)
%
% Parameter input:
%
%   'x0'        initial guess, decision variable (default zeros(N,1))
%   'w0'        initial guess, regularized decision variable (default
%               zeros(N,1))
%   'C'         regularization pre-multiplication matrix as in formula
%               (default eye(N))
%   'lam'       hyper-parameter, control strength of R (default 1.0)
%   'kap'       hyper-parameter, control strength of the quadratic penalty
%               (default 1.0)
%   'ifusenormal' use the normal equations and Cholesky factorization
%                 rather than a QR decomposition for internal least 
%                 squares solves (this has advantages for large, sparse
%                 problems)
%   'itm'       maximum number of iterations (default 100)
%   'tol'       terminate if change in w (in l2 norm) is less than tol
%               (default 1e-6)
%   'ptf'       print every ptf iterations (don't print if 0). (default 0)
%   'mode'      '2': R = 0.5*squared 2 norm, i.e. 0.5*sum(abs(x).^2)
%               '1': R = 1 norm, i.e. sum(abs(x))
%               '0': R = 0 norm, i.e. nnz(x)
%               'mixed': R = sum of 0, 1, and squared 2 norms with 
%                weights l0w, l1w, and l2w
%               'other': R and Rprox must be provided
%               (default '1')
%   'l0w'       weight of l0 norm for 'mixed' mode (default 0.0)
%   'l1w'       weight of l1 norm for 'mixed' mode (default 0.0)
%   'l2w'       weight of l2 norm for 'mixed' mode (default 0.0)
%   'R'       function evaluating regularizer R
%   'Rprox'   proximal function which, for any alpha, evaluates 
%               Rprox(x,alpha) = argmin_y alpha*R(y)+0.5*norm(x-y,2)^2
%
% output:
%   x, w the computed minimizers of the objective
%
% Example:
%
%   >> m = 100; n = 2000; k = 10;
%   >> A = randn(m,n);
%   >> y = zeros(n,1); y(randperm(n,k)) = sign(randn(k,1));
%   >> lam = A.'*b;
%   >> [x,w] = sr3(A,b,'lam',lam);
%
% See also LASSO, LINSOLVE

% Copyright 2018 Travis Askham and Peng Zheng
% Available under the terms of the MIT License

%% parse inputs

[m,n] = size(A);

[p,Rfunc,Rprox] = sr3_parse_input(A,b,m,n,varargin{:});

x = p.Results.x0;
w = p.Results.w0;
C = p.Results.C;
lam = p.Results.lam;
kap = p.Results.kap;
itm = p.Results.itm;
tol = p.Results.tol;
ptf = p.Results.ptf;
ifusenormal = p.Results.ifusenormal;
ifuselsqr = p.Results.ifuselsqr;

[md,~] = size(C);
if md ~= n
    w = zeros(md,1);
end
            
%% pre-process data

rootkap = sqrt(kap);
alpha = lam/kap;
if ifusenormal
   atareg = (A.'*A) + kap*(C.'*C);
   if issparse(atareg)
    [atacholfac,p,s] = chol(atareg,'upper','vector');
   else
    [atacholfac,p] = chol(atareg,'upper');
    s = 1:n;
   end
   if p ~= 0 
       error('error using normal equations');
   end
   atb = A.'*b;
elseif ifuselsqr
    sys = [A;rootkap*C];
    u = [b;rootkap*w];
    x = lsqr(sys,u,tol/2,100,[],[],x);    
else
    [Q,R,p] = qr([full(A);rootkap*full(C)],0);
    opts.UT = true;
end

%% start iteration

wm  = w;
err = 2.0*tol;
noi = 0;

normb = norm(b,2);

while err >= tol
    % xstep
    if ifusenormal
        u = atb + kap*(C.'*w);
        x(s) = atacholfac\(atacholfac.'\u(s));
    elseif ifuselsqr
        u = [b;rootkap*w];
        x = lsqr(sys,u,tol/2,10,[],[],x);
    else
        u = Q'*[b;rootkap*w]; % apply q* from qr 
        x(p) = linsolve(R,u,opts); % solve rx = u
    end
    
    % store C*x
    y = C*x; 
    
    % wstep
    w = Rprox(y,alpha);
    
    % update convergence information
    obj = 0.5*sum((A*x-b).^2) + lam*Rfunc(w) + 0.5*kap*sum((y-w).^2);
    err = sqrt(sum((w - wm).^2))/normb;
    wm  = w;
    
    % print information
    noi = noi + 1;
    if mod(noi, ptf) == 0
        fprintf('iter %4d, obj %1.2e, err %1.2e\n', noi, obj, err);
    end
    if noi >= itm
        break;
    end
end

end

function [p,R,Rprox] = sr3_parse_input(A,b,m,n,varargin)
%SR3_PARSE_INPUT parse the input to SR3
% Sets default values and checks types (within reason)
% See also sr3 for details

    l1R = @(x) sum(abs(x));
    l1Rprox = @(x,alpha) sign(x).*(abs(x)-alpha).*(abs(x)>alpha);

    defaultx0 = zeros(n,1);
    defaultw0 = zeros(n,1);
    defaultC = speye(n);
    defaultlam = 1.0;
    defaultkap = 1.0;
    defaultitm = 100;
    defaulttol = 1e-6;
    defaultptf = 0;
    defaultmode = '1';
    defaultl0w = 0.0;
    defaultl1w = 0.0;
    defaultl2w = 0.0;
    defaultR = l1R;
    defaultRprox = l1Rprox;
    defaultifusenormal = 0;
    defaultifuselsqr = 0;    
    
    p = inputParser;
    isdouble = @(x) isa(x,'double');
    isdoublep = @(x) isa(x,'double') && x > 0;
    isdoublepp = @(x) isa(x,'double') && x >= 0;
    isdoublem = @(x) isa(x,'double') && length(x)==m;
    isdoublen = @(x) isa(x,'double') && length(x)==n;
    isnumericp = @(x) isnumeric(x) && x > 0;
    isnumericpp = @(x) isnumeric(x) && x >= 0;    
    isfunhandle = @(x) isa(x,'function_handle');
    
    addRequired(p,'A',isdouble);
    addRequired(p,'b',isdoublem);
    addParameter(p,'x0',defaultx0,isdoublen);
    addParameter(p,'w0',defaultw0,isdouble);
    addParameter(p,'C',defaultC,isdouble);
    addParameter(p,'lam',defaultlam,isdoublep);
    addParameter(p,'kap',defaultkap,isdoublep);
    addParameter(p,'itm',defaultitm,isnumericp);
    addParameter(p,'tol',defaulttol,isdoublep);
    addParameter(p,'ptf',defaultptf,isnumericpp);
    addParameter(p,'mode',defaultmode,@ischar);
    addParameter(p,'l0w',defaultl0w,isdoublepp);
    addParameter(p,'l1w',defaultl1w,isdoublepp);
    addParameter(p,'l2w',defaultl2w,isdoublepp);
    addParameter(p,'R',defaultR,isfunhandle);
    addParameter(p,'Rprox',defaultRprox,isfunhandle);
    addParameter(p,'ifusenormal',defaultifusenormal,@isnumeric);
    addParameter(p,'ifuselsqr',defaultifuselsqr,@isnumeric);

    parse(p,A,b,varargin{:});
    
    % override if mode '0' '1' or '2' selected
    if strcmp(p.Results.mode,'0')
        l0w = 1; l1w = 0; l2w = 0;
    elseif strcmp(p.Results.mode,'1')
        l0w = 0; l1w = 1; l2w = 0;
    elseif strcmp(p.Results.mode,'2')
        l0w = 0; l1w = 0; l2w = 1;
    else
        l0w = p.Results.l0w; l1w = p.Results.l1w; l2w = p.Results.l2w;
    end

    if strcmp(p.Results.mode,'0') || strcmp(p.Results.mode,'1') ...
            || strcmp(p.Results.mode,'2') || strcmp(mode,'mixed')
        if (abs(l0w) == 0 && abs(l1w) == 0 && abs(l2w) == 0)
            warning(['all weights in mixed norm are zero', ...
                '\n prox operation does nothing'])
        end
        R = @(x) l012Rprox(x,1,l0w,l1w,l2w,0);
        Rprox = @(x,alpha) l012Rprox(x,alpha,l0w,l1w,l2w,1);
    elseif strcmp(mode,'other')
        R = p.Results.R;
        Rprox = p.Results.Rprox;
    else
        error('incorrect value for mode')
    end

    
end
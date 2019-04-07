%% Sparse Relaxed Regularized Regression (SR3) demo
%
% In this file we demonstrate the use of the |sr3| MATLAB(R) routine
% on a few examples in regularized least squares fitting.
%
% The SR3 framework is an approach to problems of the form 
%
% $$ \min_x \frac{1}{2} \|Ax-b\|_{\ell_2}^2 + \lambda R(C x) $$ 
%
% where $R(Cx)$ represents some regularizer. For example, in LASSO
% style regression, we have that $C$ is the identity and 
% $R(x)=\|x\|_{\ell_1}$.
%
% The relaxation comes in by introducing a variable $w$ which
% does a good job of minimizing the regularizer, while $x$ is allowed
% to vary a little around that. Consider instead the minimization
% problem
%
% $$ \min_{x,w} \frac{1}{2} \|Ax-b\|_{\ell_2}^2 + \lambda R(w)
%   + \frac{\kappa}{2} \|Cx-w\|_{\ell_2}^2 $$
%
% When $C$ is the identity, the variable $w$ tends to be a good
% approximation of the solution of the original problem. For sparse
% signal problems in particular, $w$ performs well at recovering
% the 'true' support. When $C$ is not the identity, the splitting
% into $x$ and $w$ allows for a simple prox-type operation to be
% applied to $w$ (for prox friendly $R$) as opposed to dealing with
% the more complicated penalty $R(Cx)$.

% initialize

clear; clf; close all;
iseed = 8675309;
rng(iseed);
set(groot, 'defaultLineMarkerSize',10)
set(groot, 'defaultLineLineWidth',2)


%% Problem 1: $\ell_1$ vs $\ell_0$ penalties
%
% In this problem we test the performance of using SR3 to 
% recover a sparse signal. Here $R$ is either the $\ell_1$ 
% or $\ell_0$ penalty. We also plot the results of a least squares
% fit and the built in |lasso| function (if available). This is a 
% relatively easy problem and all of the regularizers perform well, 
% beating the standard least squares approach for obvious reasons.

% matrix dimensions
m = 200;
n = 1000;
k = 10; % number of non-zeros in true solution
sigma = 1e-1; % additive noise

A = randn(m,n);

y = zeros(n,1);
ind = randperm(n,k);
y(ind) = sign(randn(k,1));

b = A*y+sigma*randn(m,1);

% set up parameters of fit
lam1 = 0.01; % good for l_1 regularizer
lam0 = 0.004; % good for l_0 regularizer

% apply solver
[x0, w0] = sr3(A, b, 'mode', '0', 'lam',lam0,'ptf',0);
[x1, w1] = sr3(A, b, 'lam',lam1,'ptf',0);

% built-ins
xl2 = A\b;
if exist('lasso','builtin')
    xl1 = lasso(A,b,'Lambda',lam1);
end

% plot solution
% both regularizers perform well on this problem, though the $\ell_1$
% regularizer introduces a little more bias
figure(); hold on;
plot(y, '-*b'); plot(x0, '-xr'); plot(w0, '-og'); plot(x1, '-xc');
plot(w1, '-om'); scatter(1:length(xl2),xl2,'ok', ...
    'MarkerFaceAlpha',0.25,'MarkerEdgeAlpha',0.25);
if exist('lasso','builtin')
    plot(xl1,'-ok'); 
    legend('true signal', 'x0', 'w0', 'x1', 'w1','backslash','lasso');
else
    legend('true signal', 'x0', 'w0', 'x1', 'w1','backslash');    
end

clear;

%% Problem 2: regularized derivatives
% In this problem, we take random projections of a smooth signal
% and attempt a reconstruction under a piecewise smoothness promoting
% regularization. Specifically, we assume $x$ to be a piecewise smooth
% 1D signal (though the measurements are possibly corrupted by noise). 
% We consider both $x$ given by a hat function and $x$ given by a 
% piecewise constant function. We then let $A$ be an $m \times n$ random 
% measurement matrix, with $m < n$. We set up $C$ to be the $n-1\times n$
% matrix mapping the signal to a finite difference approximation of the 
% derivative. We then reconstruct the signal as the cumulative sum of 
% $w$, adjusting for the integration constant.


iseed = 8675309;
rng(iseed);

n = 500;
m = 100;
sigma = 0.1;
A = randn(m,n);

% set up signal as step function (sparse derivative assumption)

y = zeros(n,1);
for i = 1:5
	y((i-1)*100+1:i*100) = i;
end

b = A*y + sigma*randn(m,1);

e = ones(n,1);
C = spdiags([-e,e],[0,1],n-1,n); % difference matrix

lam0 = 0.01;
lam1 = 0.02;
lam1_2 = 0.002;

% apply solver
[x0, w0] = sr3(A, b, 'mode', '0', 'lam',lam0,'ptf',0,'C',C);
[x1, w1] = sr3(A, b, 'mode', '1', 'lam',lam1,'ptf',0,'C',C);
[x1_2, w1_2] = sr3(A, b, 'mode', '1', 'lam',lam1_2,'ptf',0,'C',C);

cs = cumsum([0;w0]);
y0 = cs + (sum(x0)-sum(cs))/n;
cs = cumsum([0;w1]);
y1 = cs + (sum(x1)-sum(cs))/n;
cs = cumsum([0;w1_2]);
y1_2 = cs + (sum(x1_2)-sum(cs))/n;

figure()
plot(y,'b')
hold on
plot(y0,'--r')
plot(y1,'--g')
plot(y1_2,'--m')

legend('true signal', 'l0', 'l1 v1', 'l1 v2');    

% set up signal as piecewise linear (Chartrand example)
% note that here the regularization is on the second derivative
% also, the regularization is sensitive to lambda

n = 100;

t = linspace(0,1,n).';
tmid = (t(2:end)+t(1:end-1))/2.0;
h = t(2)-t(1);
y = abs(t-0.5);

b = y + sigma*randn(n,1);

bstart = b(1);
b = b-bstart;

sigma = 0.05;

A = tril(ones(n,n-1),-1)*h;
e = ones(n,1);
C = spdiags([-e,e],[0,1],n-2,n-1)/h;

lam0 = 0.007;
lam1 = 0.001;

xi = diff(b)/h;
wi = C*xi;

kappa0 = 1.0*h;
kappa1 = 1.0*h;

% apply solver
[x0, w0] = sr3(A, b, 'mode', '0', 'lam',lam0,'itm',10000,'ptf',0,...
    'C',C,'x0',xi,'w0',wi,'kap',kappa0);
[x1, w1] = sr3(A, b, 'mode', '1', 'lam',lam1,'itm',10000,'ptf',0,...
    'C',C,'x0',xi,'w0',wi,'kap',kappa1);


% reconstruct from w

cs = cumsum([0;w0])*h;
sw0 = cs + (sum(x0)-sum(cs))/length(cs);
sw0int = A*sw0;
coeffs = [sw0int,ones(n,1)]\(b+bstart);
y0 = coeffs(1)*sw0int + coeffs(2)*ones(n,1);
cs = cumsum([0;w1])*h;
sw1 = cs + (sum(x1)-sum(cs))/length(cs);
sw1int = A*sw1;
coeffs = [sw1int,ones(n,1)]\(b+bstart);
y1 = coeffs(1)*sw1int + coeffs(2)*ones(n,1);

figure()
plot(t,y,'b')
hold on
plot(t,b+bstart,'--','color',[0,0,0]+0.75);
plot(t,y0,'--r')
plot(t,y1,'--g')

legend('true signal','corrupted', 'l0', 'l1');

figure()
plot(tmid,diff(y),'-xb')
hold on
plot(tmid,diff(y0),'-xr')
plot(tmid,diff(y1),'-xg')
legend('true derivative', 'l0', 'l1');

%% test 1: speed of l012Rprox utility

ntest = 10000;
nx = 1001;

tgoal = 1.0;

ifprox = 1;

xs = linspace(-2,2,nx);

tic;
for i = 1:ntest
    alpha = rand(); l0w = rand()*(rand()>0.5); l1w = rand()*(rand()>0.5); 
    l2w = rand()*(rand() > 0.5);
    wprox = l012Rprox(xs,alpha,l0w,l1w,l2w,ifprox);
        
end
time = toc;

assert(time < tgoal,...
    sprintf('l012prox speed test failed, took %e s, goal: %e s',...
    time,tgoal));

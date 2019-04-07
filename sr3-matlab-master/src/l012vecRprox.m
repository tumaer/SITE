function z = l012vecRprox(x,alpha,l0w,l1w,l2w,ifprox,ndim)
%L012VECRPROX utility function for the l0, l1, and l2 penalties
% on vectors.
%
% x is assumed to be have entries which are in turn coordinates of 
% ndim-dimensional vectors. The first length(x)/ndim entries correspond
% to the first coordinate, the second length(x)/ndim entries correspond
% to the second coordinate, and so on. 
% 
% prox-ing is then applied to the euclidian norm of these points
% in ndim-dimensional space and the points are rescaled.
%
% See also L012RPROX

n2 = length(x)/ndim;
x2 = reshape(x,n2,ndim);

r = sqrt(sum(x2.^2,2));

z = l012Rprox(r,alpha,l0w,l1w,l2w,ifprox);

r = r + (r==0);

if ifprox
    z = repmat(z./r,1,ndim);
    x2 =x2.*z;

    z = x2(:);
end




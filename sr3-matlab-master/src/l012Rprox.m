
function z = l012Rprox(x,alpha,l0w,l1w,l2w,ifprox)
%L012RPROX utility function for the l0, l1, and l2 penalties
% this function returns either the value of the penalty or 
% the solution of the prox problem (if ifprox):
%
% argmin_z alpha*(l0w* nnz(z) + l1w*\|z\|_1 + l2w*0.5\|z\|_2^2) 
%                   + 0.5*\|x-z\|_2^2
%
% If ~ifprox, then 
% l0w*nnz(x) + l1w*\|x\|_1 + l2w*0.5*\|x\|_2^2 is returned
%
% input:
%   x - vector, as above
%   alpha - weight, as above
%   mode - string, '0' -> p = 0, etc.
%   ifprox - flag, as above
%

mode = (l0w ~= 0)*1 + (l1w ~= 0)*2 + (l2w ~= 0)*4;

% avoid unnecessary computation if possible
switch mode
    case 1
        if ifprox
            alpha0 = l0w*alpha;
            z = x.*(abs(x) > sqrt(2*alpha0));
        else
            z = l0w*nnz(x);
        end
    case 2
        if ifprox
            alpha1 = l1w*alpha;
            z = sign(x).*(abs(x)-alpha1).*(abs(x)>alpha1);      
        else
            z = l1w*sum(abs(x));
        end
    case 3
        if ifprox
            alpha0 = l0w*alpha;
            alpha1 = l1w*alpha;
            z = sign(x).*(abs(x)-alpha1).*(abs(x)>alpha1);
            fz = (abs(z)~=0)*alpha0+abs(z)*alpha1+0.5*abs(z-x).^2;
            z = z.*(fz < 0.5*abs(x).^2);
        else
            z = l0w*nnz(x)+l1w*sum(abs(x));
        end
    case 4
        if ifprox
            alpha2 = l2w*alpha;
            z = x/(1.0+alpha2);
        else
            z = l2w*0.5*sum(abs(x).^2);
        end
    case 5
        if ifprox
            alpha0 = alpha*l0w;
            alpha2 = alpha*l2w;
            z = x/(1+alpha2);
            fz = (abs(z)~=0)*alpha0+ ...
                0.5*alpha2*abs(z).^2+0.5*abs(z-x).^2;
            z = z.*(fz < 0.5*abs(x).^2);
        else
            z = l0w*nnz(x)+l2w*0.5*sum(abs(x).^2);
        end
    case 6
        if ifprox
            alpha1 = alpha*l1w;
            alpha2 = alpha*l2w;
            z = sign(x).*(abs(x)-alpha1).*(abs(x)>alpha1)/(1+alpha2);
        else
            z = l1w*sum(abs(x))+l2w*0.5*sum(abs(x).^2);
        end
    case 7
        if ifprox
            alpha0 = alpha*l0w;
            alpha1 = alpha*l1w;
            alpha2 = alpha*l2w;
            z = sign(x).*(abs(x)-alpha1).*(abs(x)>alpha1)/(1+alpha2);
            fz = (abs(z)~=0)*alpha0+abs(z)*alpha1+ ...
                0.5*alpha2*abs(z).^2+0.5*abs(z-x).^2;
            z = z.*(fz < 0.5*abs(x).^2);
        else
            z = l0w*nnz(x)+l1w*sum(abs(x))+l2w*0.5*sum(abs(x).^2);
        end
    otherwise
        if ifprox
            z = x;
        else
            z = 0;
        end
end

function [f,Y_min]= calculate_obj(M,XB,DM,d2_M2_jl,pars)


%Term Binary: sum_i sum_j d^2_M(x_i,x_j)
Term_B = sum(sum((XB*M).*XB))*pars.mu/pars.B;

%Term Triplet: sum_ijl [1+d^2_M(x_i,x_j)-d^2_M(x_i,x_l)]_+
Term_T = sum(max(1-DM,0))*(1-pars.mu)/pars.T;

%Term Perturbation:
Y_min  = zeros(length(DM),1);
Case   = zeros(1,3);
% Calculate the property of x_min in the objective function
% if d_M(x_i,x_l) >= d_M(x_i,x_j), Case 1, Y_min = 1;
% if d_M(x_i,x_l) <  d_M(x_i,x_j), Case 2, Y_min = 0.
% if instance in case 1 is outside the ball, Y_min = 0; Case 3, the number of active perturbation constraints
Idx     = (DM >= 0);
Y_min(Idx) = 1; Case(1) = sum(Idx);
Case(2) = length(DM) - Case(1);
d2_imin = DM.^2 /4 ./ (d2_M2_jl+pars.eps);
Idx     = find(Idx);
Y_min(Idx(d2_imin(Idx) > pars.tau^2)) = 0;
Case(3) = sum(Y_min~=0);
Term_P  = sum(pars.tau^2 - d2_imin(Y_min~=0)) + Case(2)*pars.tau^2;
Term_P  = Term_P*pars.lambda/pars.T;

f = Term_B + Term_T + Term_P;

end

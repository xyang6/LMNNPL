function [L,pars,Log] = LMNN_PL(Init,pars,Log)

tic
%% Initialisation
Triplet  = Init.Triplet;
X_imp    = Init.X_imp;
X_tar    = Init.X_tar;
X_in     = Init.X_in;
B_ij     = Init.B_ij;
T_ij     = Init.T_ij;
T_il     = Init.T_il;
T_jl     = Init.T_jl;
Term_B   = Init.Term_B;
DM       = Init.DM;
d2_M2_jl = Init.d2_jl;
pars.B   = Init.B;
pars.T   = size(Triplet,2);

iter      = 1;
M         = eye(pars.p);
max_iter  = pars.max_iter;
obj_trace = zeros(max_iter,1);
eta       = pars.eta;

[obj_trace(iter),Y_min] = calculate_obj(M,B_ij,DM,d2_M2_jl,pars);
slack_old   = (DM<1);slack_new = slack_old;

T_12   = T_ij(slack_new,:); T_13 = T_il(slack_new,:);
Term_T = T_12' * T_12 - T_13' * T_13;
clear T_12 T_13

%% Compute gradient and Update the metric
alpha_d = pars.alpha;

while iter <= max_iter
    
    %update distance metric
    [Grad,Term_T] = calculate_grad(Term_B,Term_T,T_ij,T_il,slack_old,slack_new,M,DM,d2_M2_jl,X_in,X_tar,X_imp,Y_min,pars);
    M     = M - alpha_d * (Grad+Grad')/2;
    [V,D] = eig(M);
    D(D<1e-10) = 0;
    M     = V * D * V';
    M     = (M+M')/2;
    if all(M(:)) == 0;break;end
    
    %calculate objective function and update perturbation points
    iter      = iter + 1; slack_old = slack_new;
    DM        = sum((T_il * M).*T_il,2) - sum((T_ij * M).*T_ij,2);
    d2_M2_jl  = sum((T_jl * M^2).*T_jl,2);
    [obj_trace(iter),Y_min] = calculate_obj(M,B_ij,DM,d2_M2_jl,pars);
    slack_new = (DM<1);
    
    %check convergence
    %50 iterations after active change doesn't change
    if iter >= 50 && max(abs(diff(obj_trace(iter-5:iter)))) < eta*obj_trace(iter)
        break;
    end
    
    %update learning rate
    if obj_trace(iter) < obj_trace(iter-1)
        alpha_d = alpha_d * 1.01;
    else
        alpha_d = max(alpha_d * 0.5,1e-4);
    end
    
end

L = V*sqrt(D);%M=L*L', knncl(L',...)


%% Record result
Log.M{pars.i_round}         = M;
Log.L{pars.i_round}         = L;
Log.obj_trace{pars.i_round} = obj_trace;
Log.iter(pars.i_round)      = iter;
Log.time(pars.i_round)      = toc;
if iter > max_iter
    fprintf('metric learning stops without convergence \n')
end

pars = rmfield(pars,{'mu','lambda','tau','T','B'});

end


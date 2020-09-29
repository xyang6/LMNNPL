function [Grad,Term_2,Term_B,Term_T,Term_P] = calculate_grad(Term_1,Term_2,T_ij,T_il,slack_old,slack_new,M,DM,d2_M2_jl,X_in,X_tar,X_imp,Y_min,pars)

% Binary constraint
Term_B = Term_1 *pars.mu/pars.B;

% Triplet constraint
slack = slack_new - slack_old;
Idx   = (slack==-1);
if any(Idx)
    T_12   = T_ij(Idx,:); T_13 = T_il(Idx,:);
    Term_2 = Term_2 - (T_12' * T_12 - T_13' * T_13);
end
Idx = (slack==1);
if any(Idx)
    T_12   = T_ij(Idx,:); T_13 = T_il(Idx,:);
    Term_2 = Term_2 + (T_12' * T_12 - T_13' * T_13);
end
Term_T = Term_2 *(1-pars.mu)/pars.T;

% Perturbation constraint
Term_P = zeros(pars.p);
Idx = (Y_min == 1);
if sum(Idx) ~= 0
    X_in_sub  = X_in(Idx,:);
    X_imp_sub = X_imp(Idx,:);
    X_tar_sub = X_tar(Idx,:);
    A      = DM(Idx) ./ (d2_M2_jl(Idx)+pars.eps);
    X_il   = sqrt(A/2) .* (X_in_sub  - X_imp_sub);
    X_ij   = sqrt(A/2) .* (X_in_sub  - X_tar_sub);
    X_jl   = A/sqrt(2) .* (X_tar_sub - X_imp_sub);
    Term_P = Term_P + M*(X_jl'*X_jl) - X_il'*X_il + X_ij'*X_ij;
end
Term_P = Term_P *pars.lambda/pars.T;


Grad = Term_B + Term_T + Term_P;


end

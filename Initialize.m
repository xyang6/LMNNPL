function Init = Initialize(X,Y,pars)

[Triplet,Binary] = generate_knntriplets(X,Y,pars.k,pars.v);
Triplet = Triplet';
X_imp   = X(Triplet(3,:),:);
X_tar   = X(Triplet(2,:),:);
X_in    = X(Triplet(1,:),:);
B_ij    = X(Binary(1,:),:) - X(Binary(2,:),:);
T_ij    = X_in  - X_tar;
T_il    = X_in  - X_imp;
T_jl    = X_imp - X_tar;
Term_B  = B_ij' * B_ij;

DM      = sum(T_il.^2,2) - sum(T_ij.^2,2);
d2_jl   = sum(T_jl.^2,2);

Init.Triplet = Triplet;
Init.X_imp   = X_imp;
Init.X_tar   = X_tar;
Init.X_in    = X_in;
Init.B_ij    = B_ij;
Init.T_ij    = T_ij;
Init.T_il    = T_il;
Init.T_jl    = T_jl;
Init.Term_B  = Term_B;
Init.DM      = DM;
Init.d2_jl   = d2_jl;
Init.B       = size(Binary,2);


end
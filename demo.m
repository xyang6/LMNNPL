addpath('help_functions')

clear;clc
load('data/Australian.mat')
X = data_preprocess(X);


Acc = zeros(data.N_total,1);
for i_round = 1:data.N_total
    pars.i_round = i_round;
    X_test  = X(data.Idx_test{i_round},:);
    Y_test  = Y(data.Idx_test{i_round});
    X_train = X(data.Idx_training{i_round},:);
    Y_train = Y(data.Idx_training{i_round});
    
    % set parameters
    pars.mu     = pars.Par_opt(i_round,1); %trade-off parameter in LMNN
    pars.lambda = pars.Par_opt(i_round,2); %weight of perturbation loss
    pars.tau    = pars.Par_opt(i_round,3); %desired margin
            
    % training stage
    Init_train   = Initialize(X_train,Y_train,pars);
    [L,pars,Log] = LMNN_PL(Init_train,pars,Log);
    
    % test stage
    testerr      = 1-knncl(L',X_train',Y_train',X_test',Y_test',pars.k,'train',0);
    Acc(i_round) = testerr(end);
    fprintf('round = %d, acc = %.4f \n',i_round,Acc(i_round));
    clear X_train Y_train X_test Y_test testerr L
end; clear i_round

fprintf('Final result: \nmean(acc) = %.4f, std(acc) = %.4f, running time = %.2fs\n',...
    mean(Acc),std(Acc),mean(Log.time));

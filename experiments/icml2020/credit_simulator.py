
import numpy as np
from data_prep import load_data
from strategic import best_response
from optimization import logistic_regression, evaluate_loss

# load data
np.random.seed(0) 
X, Y, data = load_data('cs-training.csv')
n = X.shape[0]
d = X.shape[1] - 1

strat_features = np.array([1, 6, 8]) - 1 # for later indexing

print('Strategic Features: \n')
for i, feature in enumerate(strat_features):
    print(i, data.columns[feature + 1])


# problems parameters
num_iters    = 25
eps_list = [.01, 1, 100, 1000]
num_eps  = len(eps_list)
params = [
    {
        'method': 'RGD',
        'opt_method': 'GD',
        'accumulation': 'OneStep'
    },
    {
        'method': 'RGD',
        'opt_method': 'GD',
        'accumulation': 'Running_SubSample',
    },
]

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
import pickle

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'Times New Roman'

results = {}

for param_setting in params:

    method     = param_setting['method']
    opt_method = param_setting['opt_method']
    accumulation = param_setting['accumulation']

    if method not in results:
        results[method] = {}
    if opt_method not in results[method]:
        results[method][opt_method] = {}
    results[method][opt_method][accumulation] = {}

    print(param_setting)
    
    # fit logistic regression model we treat as the truth
    lam = 1.0/n
    theta_true, loss_list, smoothness = logistic_regression(X, Y, lam, 'Exact')
    strat_norm = np.linalg.norm(theta_true[strat_features])

    print('Accuracy: ', ((X.dot(theta_true) > 0)  == Y).mean())
    print('Loss: ', loss_list[-1])
    print('Condition Number: ', lam / (smoothness + lam))
    print('Norm: ', np.linalg.norm(theta_true))
    print('Strat Features, Norm: ', theta_true[strat_features], strat_norm)

    theta_list         = [[np.copy(theta_true)] for _ in range(num_eps)]
    theta_gaps         = [[] for _ in range(num_eps)]
    ll_list            = [[] for _ in range(num_eps)]
    acc_list_start     = [[] for _ in range(num_eps)]
    acc_list_end       = [[] for _ in range(num_eps)]
    lp_list_start      = [[] for _ in range(num_eps)]
    lp_list_end        = [[] for _ in range(num_eps)]
    condition_num_list = [[] for _ in range(num_eps)]
    gd_cutoff_list     = [[] for _ in range(num_eps)]

    for c, eps in enumerate(eps_list):

        # initial theta
        theta = np.copy(theta_true)

        print('Running epsilon =  {}\n'.format(eps))

        running_X = [X]
        running_Y = [Y]
        
        for t in range(num_iters):
            print(t)
            
            # adjust distribution to current theta
            X_strat = best_response(X, theta, eps, strat_features)

            running_X.append(X_strat)
            running_Y.append(Y)
            
            # evaluate initial loss on the current distribution
            # performative loss value of previous theta
            loss_start = evaluate_loss(X_strat, Y, theta, lam, strat_features)
            acc = ((X_strat.dot(theta) > 0) == Y).mean()
            
            acc_list_start[c].append(acc)
            lp_list_start[c].append(loss_start)
            
            # learn on induced distribution
            theta_init = np.zeros(d+1) if opt_method == 'Exact' else np.copy(theta)
            
            if accumulation == 'Running_SubSample':
                indices  = np.random.choice(len(running_X), len(X), replace=True)
                choice_X = np.concatenate([running_X[indices[i]][i,:].reshape(1,-1) for i in range(len(X))], axis=0)
                choice_Y = np.array([running_Y[indices[i]][i] for i in range(len(X))])
                theta_new, ll, logistic_smoothness = logistic_regression(choice_X, choice_Y, lam, method=opt_method, tol=1e-7, 
                                                                    theta_init=theta_init)
            else:
                theta_new, ll, logistic_smoothness = logistic_regression(X_strat, Y, lam, method=opt_method, tol=1e-7, 
                                                                    theta_init=theta_init)

            # keep track of statistics
            ll_list[c].append(ll)
            theta_gaps[c].append(np.linalg.norm(theta_new - theta) / strat_norm)
            theta_list[c].append(np.copy(theta_new))
            
            smoothness = max(logistic_smoothness + lam, 2) # lipschitz gradient
            
            condition_num_list[c].append(lam / smoothness)
            gd_cutoff_list[c].append(lam / ((smoothness + lam) * (1 + 1.5 * smoothness)))

            # evaluate final loss on the current distribution
            loss_end = evaluate_loss(X_strat, Y, theta_new, lam, strat_features)
            acc = ((X_strat.dot(theta_new) > 0) == Y).mean()
            
            lp_list_end[c].append(loss_end)        
            acc_list_end[c].append(acc)
            
            theta = np.copy(theta_new)

    results[method][opt_method][accumulation] = {
        'theta_list': theta_list,
        'theta_gaps': theta_gaps,
        'll_list': ll_list,            
        'acc_list_start': acc_list_start,     
        'acc_list_end': acc_list_end,       
        'lp_list_start': lp_list_start,      
        'lp_list_end': lp_list_end,        
        'condition_num_list': condition_num_list, 
        'gd_cutoff_list': gd_cutoff_list     
    }

    with open('results.pickle', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
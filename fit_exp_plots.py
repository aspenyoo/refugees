def exp_decay_models(path,weights_nat,weights_judge):
    #takes in weights for nationality (grant rate every year for past 10 yrs)
    #and weights for judge (grant rate every year for past 100 dec binned by 10)
    #and fits a negative exponential model
    #as described in paper.
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    def fun_nat(params):
        # returns square loss between negative exponential and actual data

        x = np.arange(1,11)
        x.astype('float64')
        yhat = np.exp((-params[0]*(params[1]+x)))

        return np.sum((yhat-weights_nat)**2)

    def fun_judge(params):
        # returns square loss between negative exponential and actual data

        x = np.arange(1,11)
        x.astype('float64')
        yhat = np.exp((-params[0]*(params[1]+x)))

        return np.sum((yhat-weights_judge)**2)

    # get judge weights
    x0 = [1.,1.]
    params_judge = minimize(fun_judge,x0)

    # plot prediction function
    xx = np.arange(1,11)
    plt.figure()
    plt.plot(xx,weights_judge)
    plt.plot(xx,np.exp((-params_judge.x[0]*(params_judge.x[1]+xx))))

    plt.xticks(xx)
    plt.xlabel('$N$ time periods ago (10 decisions)')
    plt.ylabel('Judge ID grant rate weight')
    plt.legend(['estimated parameters','LS exponential'])
    plt.savefig(path+'/judge_weight_exp.png')


    # get nationality weights
    x0 = [1.,1.]
    params_nat = minimize(fun_nat,x0)

    # plot prediction function
    xx = np.arange(1,11)
    plt.clf()
    plt.plot(xx,weights_nat)
    plt.plot(xx,np.exp((-params_nat.x[0]*(params_nat.x[1]+xx))))
    
    
    plt.xticks(xx)
    plt.xlabel('$N$ time periods ago (years)')
    plt.ylabel('Nationality grant rate weight')
    plt.legend(['estimated parameters','LS exponential'])

    plt.savefig(path+ '/nat_weight_exp.png')







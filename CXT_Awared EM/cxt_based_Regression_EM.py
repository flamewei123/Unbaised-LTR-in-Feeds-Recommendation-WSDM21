# coding=utf-8
import sys
import math
import random
import numpy as np
import xgboost as xgb
import cPickle
sys.path.append("./conf")
import config
import pandas as pd
import time
import re, os
import gc
sys.path.append("./data_feed")
import data_feed



def train(dtrain, plst, tree_num, xgb_model,iter):  # train GBDT
    watchlist = [(dtrain, '---train')]
    ## train
    print("Start training")
    if iter == 1:
        bst = xgb.train(plst, dtrain, tree_num)
    else:
        bst = xgb.train(plst, dtrain, tree_num, xgb_model=xgb_model)

    print("Training end")
    return bst



if __name__ == "__main__":
    train_data = sys.argv[1]
    configures = {
        'train_data': train_data,
    }

    ## read the train data
    init_impr_weight = 1.0
    init_click_weight = 1.0
    init_order_weight = 1.0
    print("Reading from original train data.")
    xtrain, sku, arrid, click, pos, context, weight, group_train = \
        data_feed.read_train_data(configures['train_data'], config.schema, config.trainfeature, init_impr_weight,
                                  init_click_weight, init_order_weight)
    print("Reading data done")

    # set initial probability_decreasing way
    init_beta = []
    beta = []   # beta means the probability of an item being examed
    init_alpha = []
    alpha = []    # alpha means the interest at item from uesrs
    xi = [1.0]
    s1 = 0.99  # the first value of position bias
    for i in range(120):  # position bias
        init_beta.append(s1)
        s1 = s1 - 0.002  # 
    init_xi = 0.5    # context bias
    xi.append(init_xi)
    temp_interest = 0.15
    for i in range(len(click)):  
        init_alpha.append(temp_interest)

    # get information from data
    total_num = len(click)
    total_click = 0
    context_click = 0
    context_num = 0
    pos_click = []
    pos_total = []
    for i in range(120):  # initial pos_click, pos_total
        pos_click.append(0)
        pos_total.append(0)
    for i in range(len(click)):  # count the number of clicked items in each position or each context
        pos_total[pos[i] - 1] = pos_total[pos[i] - 1] + 1
        if context[i] == 1:
            context_num = context_num + 1
        if click[i] == 1:
            pos_click[pos[i] - 1] = pos_click[pos[i] - 1] + 1
            total_click = total_click + 1
            if context[i] == 1:
                context_click = context_click + 1

    # give init_value
    beta_l = []
    alpha = init_alpha
    beta = init_beta
    beta_l.append(beta)

    # threshold means rate of the relevant items
    tau = []
    tau.append(0.0)
    threshold = temp_interest
    tau.append(threshold)
    error = 1.0  

    # start loop
    iter = 1
    while error > 0.01:
        print(iter)
        print('current context bias:'+str(xi[len(xi)-1]))
        print('current position bias:')
        print(beta)
        print("real click rate: " + str(float(total_click) / len(click)))
        print('ratio of sampled 1: ' + str(tau[len(tau) - 1]))

        # E step
        p_E1_C0 = []
        p_R1_C0 = []
        p_context = []
        for i in range(len(click)):
            if click[i] == 1:
                p_E1_C0.append(1)
                p_R1_C0.append(1)
                if context[i] == 1:
                    p_context.append(1)
            else:
                temp_pE1 = (beta[pos[i]-1]*(1-alpha[i]*xi[len(xi)-1])) / (1-alpha[i]*xi[len(xi)-1]*beta[pos[i]-1])
                temp_pR1 = (alpha[i]*(1-beta[pos[i]-1]*xi[len(xi)-1])) / (1-alpha[i]*xi[len(xi)-1]*beta[pos[i]-1])
                if context[i] == 1:
                    temp_pCont = (xi[len(xi)-1]*(1-beta[pos[i]-1]*alpha[i])) / (1-alpha[i]*xi[len(xi)-1]*beta[pos[i]-1])
                    p_context.append(temp_pCont)
                p_E1_C0.append(temp_pE1)
                p_R1_C0.append(temp_pR1)

        E1_pos = []  # E=1 data in every pos
        E1_I1 = 0       # E=1 and I=1 data
        for i in range(120):
            E1_pos.append(0)
        for i in range(len(click)):
            E1_pos[pos[i] - 1] = E1_pos[pos[i] - 1] + p_E1_C0[i]
        for i in range(len(p_context)):
            E1_I1 = E1_I1 + p_context[i]

        # calculate cxt error
        context_SE = pow((xi[len(xi)-1]-E1_I1/context_num),2)

        # updata probablity
        xi.append(E1_I1/context_num)
        position_SE = 0.0
        new_beta=[]
        for i in range(120):
            new_beta.append(0)
        for i in range(120):
            position_SE = position_SE + pow((beta[i] - E1_pos[i] / pos_total[i]), 2)
            beta[i] = E1_pos[i] / pos_total[i]
            new_beta[i] = E1_pos[i] / pos_total[i]
        beta_l.append(new_beta)

        # calculate pos error
        error = pow((position_SE+context_SE), 0.5)
        print("current error: " + str(error))

        # Regression for interset
        # generate sample interest
        interest = []
        interest_num = 0
        for i in range(len(click)):  
            if click[i] == 1:
                interest.append(1)
                interest_num = interest_num + 1
            else:
                rand = random.random()
                if p_R1_C0[i] >= rand:
                    temp_interest = 1
                    interest_num = interest_num + 1
                else:
                    temp_interest = 0
                interest.append(temp_interest)
        threshold = float(interest_num) / len(click)  
        tau.append(threshold)
        ## xgboost parameters
        xgb_param = {
            'objective': 'binary:logistic',  
            'booster': 'gbtree',  
            'max_depth': 3,  
            'min_child_weight': 150,  
            'seed': 131,  
            'nthread': 20,  
            'subsample': 0.9,         
            'colsample_bylevel': 0.7,  
            'eta': 0.05,  
            'gamma': 0.1,  
            'silent': 0,  
            'eval_metric': 'auc',
            'early_stopping_rounds': 100,
            'save_period': 50  
        }

        print('-' * 100)
        print("xgboost params:")
        for key, value in xgb_param.iteritems():
            print("{0:<21s}: {1:<30s}".format(key, str(value)))

        plst = xgb_param.items()
        plst += [('eval_metric', 'logloss')]
        plst += [('eval_metric', 'ndcg@4-')]
        plst += [('eval_metric', 'error@0.6')]

        ## data weight when use pointwise training
        if not config.is_use_pairwise_training:  
             impr_weight = 1.0
             click_weight = 1.0
             if threshold < 0.4:
                 click_weight = 1 / threshold
             if threshold > 0.6:
                 impr_weight = 1 / (1 - threshold)
             del weight[:]
             # for train data
             for elem in interest:
                 if elem == 1:
                     weight.append(click_weight)
                 else:
                     weight.append(impr_weight)

        # generate train data
        dtrain = xgb.DMatrix(np.asarray(xtrain), label=interest, weight=weight, missing=-1.0)

        ## train
        xgb_model = None
        if iter != 1:
            xgb_model = xgb.Booster(model_file='save_cxt_model')
        bst = train(dtrain, plst, config.tree_num, xgb_model,iter)
        bst.save_model('save_cxt_model')  # save gbdt model

        # get predection from model
        preds = []
        preds = bst.predict(dtrain)

        # free dtrain
        dtrain = []

        # updata alpha
        for i in range(len(alpha)):
            alpha[i] = preds[i]

        iter = iter + 1

    # loop end
    print('*' * 30)
    print("loop end!")
    print('propensity of context bias: '+str(xi))
    print('ratio of sampled 1: ' + str(tau[len(tau) - 1]))
    print('propensity of position bias:')
    print(beta)
    #cPickle.dump(beta, open("./position_bias.pkl", "wb"))

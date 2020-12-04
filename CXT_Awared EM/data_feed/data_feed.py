import sys
import math
import numpy as np
import xgboost as xgb
import cPickle
sys.path.append("./conf")
import config
import pandas as pd
import time
import re, os
import gc


## header of the data set
header_idx = {}
for idx, fea in enumerate(config.schema):
    header_idx[fea] = idx


## Function read the data, generate feature matrix, label and weights
## input: file_path, data_format, weight
## output: each attribution
def read_train_data(train_data, schema, feat_set, init_impr_weight, init_click_weight, init_order_weight):
    print '-' * 70
    print ">> Reading the data"
    print "len(schema)  :", len(schema)
    print "len(feat_set):", len(feat_set)
    line_cnt = 0
    arrid = []
    click = []
    sku = []
    context = []
    pos = []
    weight = []
    group = []
    pre_arrival_id = ""
    cnt_cur_group = 0
    feature = []
    f = open(train_data)
    for line in f:
        if line.strip():
            line_cnt += 1
            if line_cnt < 10000 and line_cnt % 100 == 0:
                print line_cnt, "lines have been read."
            elif line_cnt % 10000 == 0:
                print line_cnt, "lines have been read."
            line_str_list = line.split("\t")
            cur_arrival_id = line_str_list[header_idx['arrival_id']]
	    # generate arrival_id
	    arrid.append(cur_arrival_id)
            if cur_arrival_id != pre_arrival_id:
                if pre_arrival_id != "":
                    group.append(cnt_cur_group)
                cnt_cur_group = 1
                pre_arrival_id = cur_arrival_id
            else:
                cnt_cur_group += 1
            # generate pos
	    position = int(line_str_list[header_idx['pos']])	    
            pos.append(position)
  
 	    # generate sku
	    cid3_brand = line_str_list[header_idx['cid3_brand']]
	    sku.append(cid3_brand)
            
	    # generate click
            ori_label = int(line_str_list[header_idx['click']])
            click.append(ori_label)
	    
  	    # generate context
	    with_similar = int(line_str_list[header_idx['with_high_similarity']])
	    context.append(with_similar)
            
	    # generate featue
            feat_list = []
            for s in line_str_list[len(schema) - len(feat_set):]:
                try:
                    feat_value = float(s)
                except:
                    feat_value = -1.0
                if math.isnan(feat_value):
                    feat_value = -1.0
                feat_list.append(feat_value)
	    if len(feat_list) != 735:   # 735 is the feature dimension
		print(len(feat_list))
            feature.append(feat_list)

            # generate weight
            temp_weight = init_impr_weight 
            if click == 1:
                temp_weight = init_order_weight * temp_weight
           
            weight.append(temp_weight)


    # the number of rows of the last arrival_id
    group.append(cnt_cur_group)
    f.close()
    
    return feature, sku, arrid, click, pos, context, weight, group



## Function read the data, generate feature matrix, label and weights
## input: file_path, data_format
## output: each attribution
def read_test_data(test_data, schema, feat_set):
    print '-' * 70
    print ">> Reading the data"
    print "len(schema)  :", len(schema)
    print "len(feat_set):", len(feat_set)
    line_cnt = 0
    action = []
    sid = []
    pos = []
    feature = []
    #price = []
    f = open(test_data)
    for line in f:
        if line.strip():
            line_cnt += 1
            if line_cnt < 10000 and line_cnt % 100 == 0:
                print line_cnt, "lines have been read."
            elif line_cnt % 10000 == 0:
                print line_cnt, "lines have been read."
            line_str_list = line.split("\t")
            # generate label
            ori_label = int(line_str_list[header_idx['action']])
            # different from read train data, here use original label
            action.append(ori_label)
            # generate sid
            sid_value = line_str_list[header_idx['sid']]
            sid.append(sid_value)
            # generate pos
            pos_value = int(line_str_list[header_idx['pos']])
            pos.append(pos_value)
            # generate featue
            feat_list = []
            for s in line_str_list[len(schema) - len(feat_set):]:
                try:
                    feat_value = float(s)
                except:
                    feat_value = -1.0
                if math.isnan(feat_value):
                    feat_value = -1.0
                feat_list.append(feat_value)
            feature.append(feat_list)

    f.close()
    return feature, sid, action, pos



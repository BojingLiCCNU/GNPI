import sys
import os
from os.path import abspath
import numpy as np
import time
import pandas as pd
from utils.generate_network import generate_network
from utils.prepare_data import prepare_Data
from utils.popphy_io import get_config, save_params, load_params
from utils.popphy_io import get_stat, get_stat_dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.SVM import GRID_TRAIN_SVM
from models.RF import RF
from models.MLPNN import MLPNN
import warnings
from datetime import datetime
from decimal import Decimal
import webbrowser
import subprocess
import json
import pickle
import warnings

CUDA_VISIBLE_DEVICES = 0
warnings.filterwarnings('ignore')
config = get_config()
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=10000000)


def train_ML_model(model_ML,addWeight_Method,txtName,config):
    #####################################################################
    # Read in Config File
    #####################################################################
    start_time = datetime.now()
    dataset = config.get('Evaluation', 'DataSet')
    k = float(config.get('GCN', 'k'))
    m = int(config.get('GCN', 'm'))
    b = int(config.get('GCN', 'b'))
    #########################################################################
    # Read in data and generate tree maps
    #########################################################################
    print("\nStarting %s on %s with %s..." % (model_ML,dataset,addWeight_Method))
    path = "../data/" + dataset
    dadhmaps, my_maps, my_benchmark, _, _, tree_features, labels, label_set, g, feature_df = prepare_Data(path,config,addWeight_Method)
    #
    if addWeight_Method !='_':
        data=dadhmaps  # 加权后的gcnH矩阵
    else:
        data=my_benchmark  # 未加权的原x组成矩阵

    num_class = len(np.unique(labels))

    if num_class == 2:
        metric = "AUC"
    else:
        metric = "MCC"

    if model_ML=='MLPNN':
        n_values = np.max(labels) + 1
        labels = np.eye(n_values)[labels]

    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, random_state=0)  # 分割数据集

    print("There are %d classes...%s" % (num_class, ", ".join(label_set)))
    #########################################################################
    # Determine which models are being trained
    #########################################################################
    stat_df = {}
    #########################################################################
    # Set up seeds for different runs
    #########################################################################

    # tree_row = X_train.shape[1]
    # tree_col = X_train.shape[2]
    # scaler = MinMaxScaler().fit(X_train.reshape(-1, tree_row * tree_col))
    # train_x = np.clip(scaler.transform(X_train.reshape(-1, tree_row * tree_col)), 0, 1).reshape(-1, tree_row * tree_col)
    # test_x = np.clip(scaler.transform(X_test.reshape(-1, tree_row * tree_col)), 0, 1).reshape(-1, tree_row * tree_col)

    scaler = MinMaxScaler().fit(X_train)
    train_x = np.clip(scaler.transform(X_train), 0, 1)
    test_x = np.clip(scaler.transform(X_test), 0, 1)

    #################################################################
    # Triain PopPhy-CNN model using tree maps
    #################################################################
    if model_ML=='SVM':
        model = GRID_TRAIN_SVM(config,np.unique(labels))
    elif model_ML =='RF':
        model = RF(config)
    elif model_ML=='MLPNN':
        # train_x=np.expand_dims(train_x,-1)
        # test_x = np.expand_dims(test_x, -1)
        model = MLPNN(train_x.shape[1],num_class,config)
    else:
        model = GRID_TRAIN_SVM(config, np.unique(labels))

    train = [train_x, y_train]
    test = [test_x, y_test]
    model.train(train)
    preds, stats, best_parame = model.test(test)
    if num_class == 2:
        stat_df["AUC"] = stats["AUC"]
    stat_df["Accuracy"] = stats["Accuracy"]
    stat_df["F1"] = stats["F1"]
    stat_df["MCC"] = stats["MCC"]
    stat_df["Precision"]= stats["Precision"]
    stat_df["Recall"] = stats["Recall"]
    stat_df["MCC"] = stats["MCC"]

    print("# %.3f" % (stats[metric]))

    # scores = popphy_model.get_feature_scores(train, g, label_set, tree_features, config)
    # for l in range(len(label_set)):
    # 	score_list = scores[:,l]
    # 	lab = label_set[l]
    # 	feature_scores[lab]["Run_" + str(run) + "_CV_" + str(fold)] = score_list

    #####################################################################
    # Save metric dataframes as files
    #####################################################################

    end_time = datetime.now()
    time_use = str((end_time - start_time).seconds)

    # 打印结果
    print("-------------------------------------")
    if addWeight_Method == 'Height':
        print('k=', k, ' m=', m, ' b=', b)
    elif addWeight_Method !='_':
        print('k=',k,' m=',m)
    else:
        pass
    print(stat_df)
    print("best_parame: \n", best_parame)
    txtresult = open(txtName, "a+")
    if addWeight_Method == 'Height':
        txtresult.write('m='+str(m)+' k='+str(k) +' b='+str(b)+'\n')
    elif addWeight_Method !='_':
        txtresult.write('m=' + str(m) + ' k=' + str(k) + '\n')
    else:
        pass
    txtresult.write(str(best_parame)+'\n')
    txtresult.write("time_use=" + time_use + "\n")
    txtresult.write(str(stat_df)+'\n')
    txtresult.write("--------------------------------------------\n")

    # txtresult = open ( txtName , "a" )
    # txtresult.write(str(stat_df.mean(1)))
    # txtresult.write("k="+str(k)+"\n m="+str(m))

#####################################################################
# Build Single Final Predictive Model
#####################################################################
# final_x = MinMaxScaler().fit_transform(my_maps.reshape(-1, tree_row * tree_col)).reshape(-1,tree_row, tree_col)
# train = [final_x, labels_oh]

# train_weights = []
#
# for l in np.unique(labels):
# 	a = float(len(labels))
# 	b = 2.0 * float((np.sum(labels==l)))
# 	c_prob[int(l)] = a/b
#
# c_prob = np.array(c_prob).reshape(-1)
#
# for l in np.argmax(labels_oh, 1):
# 	train_weights.append(c_prob[int(l)])
# train_weights = np.array(train_weights)
# print("Building final predictive model...")
# popphy_model = PopPhyCNN((tree_row, tree_col), num_class, config)
# popphy_model.train(train, train_weights)
# popphy_model.model.save(result_path + '/PopPhy-CNN.h5')


if __name__ == "__main__":
    # train_PopPhy(k=1,m=5)
    # config里面看数据集、DROPOUT(默认0.3)
    # prepare_data上半部分里面看哪种加权
    # prepare_data中间部分看文件名，哪种扩充方式，扩充了几倍
    config = get_config()
    model_ML='SVM'  # SVM/RF/MLPNN
    addWeight_Method='Children'  # Height\Patri\Children\'_'  '_'代表x不加权，仅取原丰富数x
    dataset = config.get('Evaluation', 'DataSet')
    ticks = time.strftime("%m%d_%H%M", time.localtime())
    txtName = str(dataset) + str(ticks) + '_'+model_ML + '_'+addWeight_Method+'_result.csv'
    if addWeight_Method =='Height':
        for b in range(5,15,5):
            config.set('GCN', 'b', str(b))
            for m in range(3,9,2):
                config.set('GCN','m',str(m))
                for k in np.arange(Decimal('-1.0'),Decimal('-0.0'),Decimal('0.1')):
                    config.set('GCN', 'k', str(k))
                    train_ML_model(model_ML,addWeight_Method,txtName,config)
    elif addWeight_Method =='Children':
        for m in range(0, 9, 2):
            config.set('GCN', 'm', str(m))
            for k in np.arange(Decimal('0.2'), Decimal('3.2'), Decimal('0.1')):
                config.set('GCN', 'k', str(k))
                train_ML_model(model_ML, addWeight_Method, txtName, config)
    elif addWeight_Method =='Patri':
        for m in range(-4, 12, 2):
            config.set('GCN', 'm', str(m))
            for k in np.arange(Decimal('-1.0'), Decimal('3.2'), Decimal('0.2')):
                config.set('GCN', 'k', str(k))
                train_ML_model(model_ML, addWeight_Method, txtName, config)
    else:
        train_ML_model(model_ML, addWeight_Method, txtName, config)
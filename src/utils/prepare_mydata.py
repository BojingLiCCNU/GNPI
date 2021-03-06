import numpy as np
import os
import struct
from array import array as pyarray
from numpy import unique
from utils.graph import Graph
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import pickle
from copy import deepcopy
import numpy.matlib

num_cores = multiprocessing.cpu_count()


# Convert abundance vector into tree matrix
def generate_maps(x, g, f, k, m, addWeight_Method, normaladj):
    # x,g,features_df    for x in data.values
    # x=遍历的第一个样本。datavalue从第一行即第一个样本遍历到最后一行最后一个样本，再把第一行转置。即一个269个元素的列，代表一个样本
    # f=微生物名字表
    id = multiprocessing.Process()._identity
    temp_g = deepcopy(g)
    temp_g.populate_graph(f, x)

    # ----------------------------------------------------
    # 在这里加权是已经构建了原版的有丰度的树以后，加权
    # map 为 树形化的加权/不加权矩阵，hmatrix 为加权/不加权的向量（GCN中的H）
    if addWeight_Method == 'Children':
        temp_g.addWeight_Children(k, m)  # 对树节点加权
    elif addWeight_Method == 'Height':
        temp_g.addWeight_Height(k, m)
    elif addWeight_Method == 'Patri':
        temp_g.addWeight_Patri(k, m)
    else:  # 此处均不加权
        pass
    map = temp_g.get_map()  # 树形矩阵
    vector = temp_g.graph_vector()  # 按树遍历的向量
    hmatrix = np.array(vector)  # 即GCN中的H矩阵
    dadhmaps = np.matmul(normaladj, hmatrix)  # gcn中的DAD*H的结果
    # ----------------------------------------------------
    del (temp_g)
    return x, np.array(map), np.array(vector), hmatrix, dadhmaps


def get_feature_df(features):
    # 每一种微生物名字剥离出来，分门别类，比如一行知道他是哪个界，哪个门，这样排序，成一个表。
    # 每一行都是一种微生物（即特征）所属的界、门、科、目..
    kingdom, phylum, cl, order, family, genus, species = [], [], [], [], [], [], []
    for f in features:

        name = f.split("k__")[1].split("|p__")[0].replace(".", "")
        if "_unclassified" in name:
            name = 'unclassified_' + name.split("_unclassified")[0]
        kingdom.append(name)

        if "p__" in f:
            name = f.split("p__")[1].split("|c__")[0].replace(".", "")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                phylum.append(name)
            else:
                phylum.append("NA")
        else:
            phylum.append("NA")

        if "c__" in f:
            name = f.split("c__")[1].split("|o__")[0].replace(".", "")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                cl.append(name)
            else:
                cl.append("NA")
        else:
            cl.append("NA")

        if "o__" in f:
            name = f.split("o__")[1].split("|f__")[0].replace(".", "")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                order.append(name)
            else:
                order.append("NA")
        else:
            order.append("NA")

        if "f__" in f:
            name = f.split("f__")[1].split("|g__")[0].replace(".", "")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                family.append(name)
            else:
                family.append("NA")
        else:
            family.append("NA")

        if "g__" in f:
            name = f.split("g__")[1].split("|s__")[0].replace(".", "")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                genus.append(name)
            else:
                genus.append("NA")
        else:
            genus.append("NA")

        if "s__" in f:
            name = f.split("s__")[1]
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                species.append(name)
            else:
                species.append("NA")
        else:
            species.append("NA")

    if len(species) == 0:
        d = {'kingdom': kingdom, 'phylum': phylum, 'class': cl,
             'order': order, 'family': family, 'genus': genus}
        feature_df = pd.DataFrame(data=d)
        feature_df.index = feature_df['genus']
    else:
        d = {'kingdom': kingdom, 'phylum': phylum, 'class': cl,
             'order': order, 'family': family, 'genus': genus, 'species': species}
        feature_df = pd.DataFrame(data=d)
        feature_df.index = feature_df['species']
    return feature_df


def filter_data(x, y, core_thresh, opp_thresh):
    classes = np.unique(y)
    # 返回去掉重复字符的数组
    index = x.index.values

    core = pd.DataFrame(index=index)
    transient = pd.DataFrame(index=index)
    oppurtunistic = pd.DataFrame(index=index)

    num_counts = {}

    for c in classes:
        sub_x = x.loc[y == c]
        num_samples = len(sub_x)
        num_counts[str(c)] = sub_x[sub_x > 0].count() / float(num_samples)

    for feat in x.columns.values:
        for c in classes:
            if (num_counts[str(c)].loc[feat] >= core_thresh):
                core[feat] = x[feat]
                break

    return core


def prepare_Data(path, config, k, m, addWeight_Method):
    thresh = config.get('Evaluation', 'FilterThresh')

    # 542行(微生物)，232列（样本），第一列为名称，后为数据，
    data = pd.read_csv(path + '/abundance.tsv', index_col=0, sep='\t', header=None)

    # 一行，232列，依次记录"n"和"Cirrhosis"
    labels = np.genfromtxt(path + '/labels.txt', dtype=np.str_, delimiter=',')

    core_filt_thresh = float(thresh)
    opp_filt_thresh = 0.0

    # 此时542列微生物特征和232行样本
    data = data.transpose()

    # 232个样本，每个样本的各类微生物之和，均为100
    sums = data.sum(axis=1)
    data = data.divide(sums, axis=0)

    # label_set=['n','Cirrhosis']
    # labels:一行，前114个为0，后118个为1
    labels, label_set = pd.factorize(labels)

    # pos:118行，neg:114行
    pos_set = data.iloc[np.where(labels == 1)]
    neg_set = data.iloc[np.where(labels == 0)]

    # 可能是过滤数据或者是打乱数据顺序，原本232个样本，542个微生物特征，经过filter以后为232个样本，269个微生物特征
    core = filter_data(data, labels, core_filt_thresh, opp_filt_thresh)

    data = core

    features = list(data.columns.values)
    print("There are %d raw features..." % (len(features)))

    # 每一种微生物名字剥离出来，分门别类，比如一行知道他是哪个界，哪个门，这样排序，成一个表。
    # 看做微生物名字表：每一行都是一种具体微生物（即特征）所属的界、门、科、目、纲、属、种
    features_df = get_feature_df(features)

    print("Building tree structure...")
    try:
        g = pickle.load(open(path + "/PopPhy-tree-" + str(core_filt_thresh) + "-core.pkl", 'rb'))
        print("Found tree file...")
    except:
        print("Tree file not found...")
        print("Contsructing tree..")
        g = Graph()
        g.build_graph()
        g.prune_graph(features_df)

        # build_graph为根据很多括号的通用树文件建立的树
        # 而features_df为单一数据集中出现的微生物特征，根据当前数据集实际微生物特征修剪通用的进化树。
        g.removeRepeatName()

        # GCN中DAD
        normaladj = g.getgcninfo()

        g.routeToRoot()

    # pickle.dump(g, open(path + "/PopPhy-tree-" + str(core_filt_thresh) + "-core.pkl", 'wb'))

    print("Populating trees...")

    # data.values 是232行,每一行一个样本。269列，每一列一个微生物特征的纯数据，不带名字
    # x 为data从第一行即第一个样本遍历到最后一行最后一个样本，再把第一行转置。即一个269个元素的列，代表一个样本
    results = Parallel(n_jobs=num_cores)(
        delayed(generate_maps)(x, g, features_df, k, m, addWeight_Method, normaladj) for x in data.values)

    # dadhlist = []
    # for i in range(0,len(results)):
    # 	hmatrix = results[i][3]
    # 	hmatrix = np.matmul(normaladj,hmatrix)
    # 	dadhlist.append(hmatrix)

    my_maps = np.array(np.take(results, 1, 1).tolist())
    dadhmaps = np.array(np.take(results, 4, 1).tolist())
    # counts = np.count_nonzero(my_maps, axis=0)

    my_benchmark = np.array(np.take(results, 0, 1).tolist())  # 为加任何修饰的向量
    my_benchmark_tree = np.array(np.take(results, 2, 1).tolist())  #

    tree_features = g.graph_vector_features()

    my_benchmark_df = pd.DataFrame(index=tree_features, data=np.transpose(my_benchmark_tree))
    my_benchmark_df = my_benchmark_df.groupby(my_benchmark_df.index).mean()

    tree_features = my_benchmark_df.index
    my_benchmark_tree = np.transpose(my_benchmark_df.values)

    num_tree_features = len(tree_features)
    print("There are %d tree features..." % (num_tree_features))
    return dadhmaps, my_maps, my_benchmark, my_benchmark_tree, features, tree_features, labels, label_set, g, features_df


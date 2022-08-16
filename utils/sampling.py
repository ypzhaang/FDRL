#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0

    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
             if len(idxs_dict[label]) < 5000:
                idxs_dict[label].append(i)#字典，类别：[数据集属于该类的ID]
                count += 1#count=60000



    shard_per_class = int(shard_per_user * num_users / num_classes)#  (2*50)/10
    #count = 25000
    samples_per_user = int( count/num_users )#60000/50
    # whether to sample more test samples per user
    if (samples_per_user < 100):#无需更多数据
        double = True
    else:
        double = False

    for label in idxs_dict.keys():#lable=1---10中某一类的字典
        x = idxs_dict[label]#list数据，lable类中的数据ID
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))#换成shard_per_class行
        x = list(x)#10行

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])#拼接
        idxs_dict[label] = x        #lable：[ 10个 array]

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class #10个0 -- 9的列表
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))  #50,2的矩阵，每个user对应的两类

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)#从len中,不重复采样
            if (samples_per_user < 100 ):#and testb
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)#id:data

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])#去除数组中的重复数字，并进行排序之后输出。
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all

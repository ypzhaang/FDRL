
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate
from models.test import test_img_local_all
import os

import time

if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"]='1'
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #100个用户
    lens = np.ones(args.num_users)

    if 'cifar' in args.dataset or 'mnist' in args.dataset:#   args.dataset == 'mnist' or args.dataset == 'fashionmnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        print(len(dataset_train))
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])

    else:
        if 'femnist' in args.dataset:
            train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
            test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
        else:
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys()) 
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    print("algorithm："+args.alg)

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))#用于将预训练的参数权重加载到新的模型之中

    total_num_layers = len(net_glob.state_dict().keys())#state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
    print("total_num_layers:")
    print(total_num_layers)
    print("net_glob.state_dict().keys()")
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]
    print("*net_glob.state_dict().keys()")
    print(*net_glob.state_dict().keys())

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)指定表示参数（在 w_glob_keys 中）和头部参数（所有其他）
    if args.alg == 'fdrl' or args.alg == 'fedper':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [5,6,7,8]]
            w_local_keys = [net_glob.weight_keys[i] for i in [0,1,3,4]]
        elif 'mnist' or 'fashionmnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [3,4,5]]#[0,1,2]
            w_local_keys = [net_glob.weight_keys[i] for i in [0,1,2]]#[0,1,2]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1,2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2,3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,6,7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox':
        w_glob_keys = []
        w_local_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))#去除内嵌
        w_local_keys = list(itertools.chain.from_iterable(w_local_keys))#去除内嵌
    
    print(total_num_layers)
    print(w_glob_keys)
    print(net_keys)
    if args.alg == 'fdrl' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
            #print(w_local_dict[key].shape)
        w_locals[user] = w_local_dict   #id:{各层：参数}


    indd = None      # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    for iter in range(args.epochs+1):#每一轮
        w_glob = {}
        loss_locals = []
        m = int(args.frac * args.num_users)#m = max(int(args.frac * args.num_users), 10)#10#
        if iter == args.epochs:
            m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 从用户数中随机抽m个用户
        print("idxs_users:")
        print(idxs_users)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len=0
        for ind, idx in enumerate(idxs_users):#对每一位用户
            start_in = time.time()
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd)
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd)
            else:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_local = copy.deepcopy(net_glob)###
            w_local = net_local.state_dict()
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            last = iter == args.epochs
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                w_local, loss, indd = local.train(net=net_local.to(args.device),ind=idx, idx=clients[idx], w_glob_keys=w_glob_keys, w_local_keys=w_local_keys, lr=args.lr,last=last)
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys, w_local_keys=w_local_keys, lr=args.lr, last=last)

            loss_locals.append(copy.deepcopy(loss))
            
            total_len += lens[idx]
            if len(w_glob) == 0:#每个客户端自己对应的global
                w_glob = copy.deepcopy(w_local)
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key]*lens[idx]
                    else:
                        w_glob[key] += w_local[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]

            times_in.append( time.time() - start_in )


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        print("local loss:")
        print(loss_locals)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        w_local = net_glob.state_dict()#sever分发下去
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_glob_keys=w_glob_keys, w_locals=w_locals,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:#  ?  -10
                accs10 += acc_test/10

            if iter >= args.epochs-10 and iter != args.epochs:
                accs10_glob += acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    print("loss:")
    print(loss_train)

    end = time.time()
    print(end-start)
    print(times)
    print("acc:")
    print(accs)
    base_dir = './save/accs_' + args.alg + '_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)

import numpy as np
from sklearn.utils import shuffle
import torch
import random
import pandas as pd
import pickle
from collections import defaultdict
import itertools
from dataset import GDataset
# from models.LightGCN_hete import LightGCN
# from models.MF import MF
from models.IGRec import IGRec
from models.Predictor import HeteroDotProductPredictor
from utils.util import Helper
from utils.parser import parse_args
import pdb
from torch.utils.data import TensorDataset, DataLoader
import torch as t
import dgl
# import dgl.function as fn
import scipy.sparse as sp
import logging


def innerProduct(u, i, j):
    pred_i = t.sum(t.mul(u,i), dim=1)
    pred_j = t.sum(t.mul(u,j), dim=1)
    return pred_i, pred_j

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data(dataset):
    edge_src_1, edge_dst_1 = dataset.user_trainMatrix.nonzero()
    data_dict = dict()
    data_dict[('user', 'rate', 'item')] = (edge_src_1, edge_dst_1)
    data_dict[('item', 'rated by', 'user')] = (edge_dst_1, edge_src_1)
    num_nodes_dict = {'user':dataset.num_users, 'item':dataset.num_items}
    graph = dgl.heterograph(data_dict, num_nodes_dict= num_nodes_dict)
    ratings = torch.tensor([1.0 for i in range(len(edge_src_1))])
    graph.edges['rate'].data['y'] = ratings
    graph.edges['rated by'].data['y'] = ratings
    val = dataset.user_valRatings
    return graph, val


def build_val_graph(validation, graph):
    validation = torch.tensor(validation)
    src = validation[:,0]
    dst = validation[:,1]
    data_dict = dict()
    data_dict[('user', 'rate', 'item')] = (src, dst)
    data_dict[('item', 'rated by', 'user')] = (dst, src)
    num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}
    val_graph = dgl.heterograph(data_dict, num_nodes_dict= num_nodes_dict)

    return val_graph

def construct_negative_graph(graph, etype,device):
    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape).to(device)
    graph_neg =dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})
    return graph_neg

def training(model,group_user):
    best_val_loss = 99999999
    stop_count = 0
    for e in range(args.epochs):
        print('Epoch: ', e)
        model.train()
        graph_neg = construct_negative_graph(uv_g, ('user', 'rate', 'item'),device)
        user_embed, item_embed, group_emb = model(uv_g,group_user)
        h= {"user": user_embed, "item": item_embed}
        score = predictor(uv_g, h, ('user', 'rate', 'item'))
        score_neg = predictor(graph_neg, h, ('user', 'rate', 'item'))

        bprloss = -(score - score_neg).sigmoid().log().sum()
        if args.num_aspects > 1:
            aspect_reg_loss = model.aspect_regular()
        else:
            aspect_reg_loss = 0
        aspect_reg_loss = 0
        # loss = bprloss
        h2 = {"group": group_emb, "item": model.item_embedding}
        group_score = predictor(graph_g_i, h2, ('group', 'rate', 'item'))
        group_graph_neg = construct_negative_graph(graph_g_i, ('group', 'rate', 'item'),device)
        group_neg_score = predictor(group_graph_neg, h2, ('group', 'rate', 'item'))
        group_loss = -(group_score - group_neg_score).sigmoid().log().sum()

        loss = (1 - args.group_loss_reg) * bprloss + args.reg_coef * aspect_reg_loss + args.group_loss_reg * group_loss
        # loss = (1 - args.group_loss_reg) * bprloss +  args.group_loss_reg * group_loss
        # pdb.set_trace()
        # loss = (1.0 - args.reg_coef) * bprloss + args.reg_coef * aspect_reg_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.empty_cache()
        #calculate val loss for early stop

        with torch.no_grad():
            model.eval()

            user_embed, item_embed, group_emb = model(uv_g,group_user)

            val_loss = 0
            h= {"user": user_embed, "item": item_embed}
            h2 = {"group": group_emb, "item": model.item_embedding}
            graph_neg = construct_negative_graph(val_graph, ('user', 'rate', 'item'),device)
            score = predictor(val_graph, h, ('user', 'rate', 'item'))
            score_neg = predictor(graph_neg, h, ('user', 'rate', 'item'))
            if args.num_aspects > 1:
                aspect_reg_loss = model.aspect_regular()
            else:
                aspect_reg_loss = 0
            # aspect_reg_loss = 0
            group_score = predictor(graph_g_i_val, h2, ('group', 'rate', 'item'))
            group_graph_val_neg = construct_negative_graph(graph_g_i_val, ('group', 'rate', 'item'),device)
            group_neg_score = predictor(group_graph_val_neg, h2, ('group', 'rate', 'item'))
            group_loss = -(group_score - group_neg_score).sigmoid().log().sum()
            # val_loss = (-(score - score_neg).sigmoid().log().sum())/val_length
            
            bprloss = -(score - score_neg).sigmoid().log().sum()
            # aspect_reg_loss = model.aspect_regular()
            # val_loss = (1 - args.group_loss_reg) * bprloss +  args.group_loss_reg * group_loss
            val_loss = (1 - args.group_loss_reg) * bprloss + args.reg_coef * aspect_reg_loss + args.group_loss_reg * group_loss
            # val_loss = (1.0 - args.reg_coef) * bprloss + args.reg_coef * aspect_reg_loss
            print("val loss: ", val_loss)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_count = 0
            torch.save(model, 'models/saved_model/' + args.data+'_'+str(args.lr)+'_'+str(args.weight_decay)+ '_' +str(args.embed_size) +'_' +str(args.layer_num) +'_'+str(args.num_aspects)+ '_'+str(args.threshold)+'_'+str(args.reg_coef) + '_'+ str(args.group_loss_reg)+'_' + str(args.tau_gumbel) +'_'+ str(args.seed) +'.pt')
        else:
            stop_count += 1
            if stop_count > args.early_stop:
                break

    model_final = torch.load('models/saved_model/' + args.data+'_'+str(args.lr)+'_'+str(args.weight_decay)+ '_' +str(args.embed_size) +'_' +str(args.layer_num) +'_'+str(args.num_aspects)+ '_'+str(args.threshold)+'_'+str(args.reg_coef) + '_'+ str(args.group_loss_reg)+'_' + str(args.tau_gumbel) +'_'+ str(args.seed) +'.pt')
    model_final.cpu()
    model_final.eval()
    # graph_g_u_p = graph_g_u_p.to(torch.device('cpu'))
    group_user = group_user.to(torch.device('cpu'))
    graph = uv_g.to(torch.device('cpu'))
    result = helper.test_model(model_final, u_i_train_dict,u_i_test_dict, K, graph,group_user, target='user')
    print(result)
    print('\t')
    group_graph = graph_g_i.to(torch.device('cpu'))
    group_result = helper.test_model(model_final, g_i_train_dict,g_i_test_dict, K, graph,group_user, target='group',g_i_graph = group_graph)
    print(group_result)

def build_side_graph(dataset,g_m_d):
    edge_g2, edge_u = [],[]
    for g,u_list in g_m_d.items():
        for u in u_list:
            edge_g2.append(g)
            edge_u.append(u)
    data_dict2 = dict()
    data_dict2[('group', 'contain', 'user')] = (edge_g2, edge_u)
    data_dict2[('user', 'in', 'group')] = (edge_u, edge_g2)
    num_nodes_dict2 = {'group':len(g_m_d), 'user':dataset.num_users}
    graph_g_u = dgl.heterograph(data_dict2, num_nodes_dict= num_nodes_dict2)
    ratings = torch.tensor([1.0 for i in range(len(edge_g2))])
    graph_g_u.edges['contain'].data['y'] = ratings
    graph_g_u.edges['in'].data['y'] = ratings

    return  graph_g_u

def save_graph(G,path,name):
    with open(path+name, 'wb') as file:
        pickle.dump(G, file)

def open_graph(path,name):
    with open(path+name, 'rb') as file:
        N = pickle.load(file)
    return N


if __name__ =='__main__':
    # setup_seed(1008)
    args = parse_args()
    print(args)
    setup_seed(args.seed)

    device = torch.device('cuda:'+str(args.cuda)  if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    data_path = './datasets/'+ args.data+'/'
    predictor = HeteroDotProductPredictor()
    helper = Helper()
    if args.pre == 0:
        dataset = GDataset(data_path, args.num_negatives)
        uv_g, val= prepare_data(dataset)
        save_graph(uv_g,data_path,'uv_g')
        graph_g_i = dataset.graph_g_i
        graph_g_i_val = dataset.graph_g_i_val
        save_graph(dataset.graph_g_i,data_path,'graph_g_i')
        save_graph(dataset.graph_g_i_val,data_path,'graph_g_i_val')
        g_m_d = helper.gen_group_member_dict('./datasets/' + args.data + '/groupMember.txt')
        num_group = len(g_m_d)

        group_user = torch.zeros(num_group,dataset.num_users)
        for g,u_list in g_m_d.items():
            members = torch.LongTensor(u_list)
            group_user[g][members] = 1

        torch.save(group_user, data_path+'group_user.pt')
        graph_g_u = build_side_graph(dataset,g_m_d)

        save_graph(graph_g_u,data_path,'graph_g_u')
        val_graph = build_val_graph(val,uv_g)

        save_graph(val_graph,data_path,'val_graph')
        save_graph(dataset.u_i_train_dict,data_path,'u_i_train_dict')
        save_graph(dataset.u_i_test_dict,data_path,'u_i_test_dict')
        save_graph(dataset.g_i_train_dict,data_path,'g_i_train_dict')
        save_graph(dataset.g_i_test_dict,data_path,'g_i_test_dict')
        u_i_train_dict = dataset.u_i_train_dict
        u_i_test_dict = dataset.u_i_test_dict
        g_i_train_dict = dataset.g_i_train_dict
        g_i_test_dict = dataset.g_i_test_dict
    else:
        uv_g = open_graph(data_path,'uv_g')
        graph_g_u = open_graph(data_path,'graph_g_u')
        val_graph = open_graph(data_path,'val_graph')
        graph_g_i = open_graph(data_path,'graph_g_i')
        graph_g_i_val = open_graph(data_path,'graph_g_i_val')
        group_user = torch.load(data_path+'group_user.pt')
        u_i_train_dict = open_graph(data_path,'u_i_train_dict')
        u_i_test_dict = open_graph(data_path,'u_i_test_dict')
        g_i_train_dict = open_graph(data_path,'g_i_train_dict')
        g_i_test_dict = open_graph(data_path,'g_i_test_dict')

    
    group_user = group_user.to(device)

    uv_g = uv_g.to(device)
    graph_g_i = graph_g_i.to(device)
    graph_g_i_val = graph_g_i_val.to(device)
    graph_g_u = graph_g_u.to(device)
    val_graph = val_graph.to(device)

    # val_length = val_graph.num_edges()

    model = IGRec(args,graph_g_u,uv_g,device)
    model = model.to(device)
    opt = t.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    # K = args.topK
    K = [5,10,15,20,50,100]
    training(model,group_user)




import pdb
import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
import dgl
from collections import defaultdict

class GDataset(object):
    def __init__(self, path,num_negatives):
    # def __init__(self, user_path, group_path, num_negatives):
        '''
        Constructor
        '''
        self.data_size(path)
        self.num_negatives = num_negatives
        # user data
        self.user_valRatings,self.u_i_val_dict = self.load_rating_file_as_list(path + "val_ui.txt")
        self.user_testRatings, self.u_i_test_dict= self.load_rating_file_as_list(path + "test_ui.txt")
        self.user_trainMatrix, self.u_i_train_dict = self.load_rating_file_as_matrix(path + "train_ui.txt", 'user')
    
        # group data
        self.group_testRatings,self.g_i_test_dict = self.load_rating_file_as_list(path + "test_gi.txt")
        self.group_valMatrix, self.g_i_val_dict = self.load_rating_file_as_matrix(path + "val_gi.txt", 'group')
        self.group_trainMatrix, self.g_i_train_dict = self.load_rating_file_as_matrix(path + "train_gi.txt", 'group')
        self.graph_g_i = self.integrate_group_item_dataset(self.group_trainMatrix)
        self.graph_g_i_val = self.integrate_group_item_dataset(self.group_valMatrix)
        # self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")
    def data_size(self, path):
        with open(path +'/data_size.txt') as f:
            line = f.readline()
            elements = line.split(' ')
            self.num_users, self.num_items, self.num_groups = int(elements[0]), int(elements[1]),int(elements[2])
        f.close()
        
    def integrate_group_item_dataset(self, matrix):

        group, item = matrix.nonzero()
        data_dict = dict()
        data_dict[('group', 'rate', 'item')] = (group, item)
        data_dict[('item', 'rated by', 'group')] = (item, group)
        num_nodes_dict = {'group':self.num_groups, 'item':self.num_items}
        graph = dgl.heterograph(data_dict, num_nodes_dict= num_nodes_dict)
        ratings = torch.tensor([1.0 for i in range(len(group))])
        graph.edges['rate'].data['y'] = ratings
        graph.edges['rated by'].data['y'] = ratings
        return graph

    def load_rating_file_as_list(self, filename):
        u_i_dict = {}
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])

                if user not in u_i_dict:
                    u_i_dict[user] = [item]
                else:
                    u_i_dict[user].append(item)
                ratingList.append([user, item])
                line = f.readline()
        return ratingList, u_i_dict

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename, type):
        # Get number of users and items
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                line = f.readline()
        # Construct matrix
        if type =='user':
            mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        if type =='group':
            mat = sp.dok_matrix((self.num_groups, self.num_items), dtype=np.float32)
        u_i_dict = defaultdict(list)

        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                        u_i_dict[user].append(item)
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                    u_i_dict[user].append(item)
                line = f.readline()
        return mat, u_i_dict

    def get_train_instances(self,mat):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = mat.shape[0]
        num_items = mat.shape[1]
        
        u_p_n_dict = dict()
        for (u, i) in mat.keys():
            u_p_n_dict[(u,i)] = []
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in mat:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
                u_p_n_dict[(u,i)].append(j)

        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self,batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader

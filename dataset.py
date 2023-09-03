import pdb
import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data

class GDataset(object):
    def __init__(self, user_path, num_negatives):
    # def __init__(self, user_path, group_path, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives
        # user data
        self.user_valRatings,self.u_i_val_dict,num_user_val, num_items_val = self.load_rating_file_as_list(user_path + "val_ui.txt")
        self.user_testRatings, self.u_i_test_dict, num_user_test, num_items_test = self.load_rating_file_as_list(user_path + "test_ui.txt")
        current_max_user = max(num_user_val,num_user_test)
        current_max_item = max(num_items_val,num_items_test)
        self.user_trainMatrix, self.u_i_train_dict = self.load_rating_file_as_matrix(user_path + "train_ui.txt", current_max_user, current_max_item)
        self.num_users, self.num_items = self.user_trainMatrix.shape
    


    def load_rating_file_as_list(self, filename):
        u_i_dict = {}
        ratingList = []
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                num_users = max(num_users, user)
                num_items = max(num_items, item)
                if user not in u_i_dict:
                    u_i_dict[user] = [item]
                else:
                    u_i_dict[user].append(item)
                ratingList.append([user, item])
                line = f.readline()
        return ratingList, u_i_dict, num_users, num_items

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

    def load_rating_file_as_matrix(self, filename, n_u, n_i):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        num_users = max(num_users, n_u)
        num_items = max(num_items, n_i)
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        u_i_dict = {}
        for i in range(num_users+1):
            u_i_dict[i] = []

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

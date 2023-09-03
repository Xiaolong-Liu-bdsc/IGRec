from torch.autograd import Variable
import numpy as np
import math
import heapq
import torch as t
import pdb
from torch.utils.data import TensorDataset, DataLoader
import tqdm

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d

    def evaluate_model(self, model, testRatings, testNegatives, K, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []

        for idx in range(len(testRatings)):
            (hr,ndcg) = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
            hits.append(hr)
            ndcgs.append(ndcg)
        return (hits, ndcgs)



    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = t.from_numpy(users)
        users_var = users_var.long()
        items_var = t.LongTensor(items)
        if type_m == 'group':
            predictions = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0

    def test_model(self, model, u_i_train_dict,u_i_test_dict, K,uv_g,group_user):
        users = []
        num_users = uv_g.nodes('user').shape[0]
        num_items = uv_g.nodes('item').shape[0]
        all_train = t.zeros(num_users,num_items)
        all_test = t.zeros(num_users,num_items)
        

        for u,i_list in u_i_train_dict.items():
            item_train = t.LongTensor(i_list)
            all_train[u][item_train] = 1

        for u,i_list in u_i_test_dict.items():
            users.append(u)
            item_test = t.LongTensor(i_list)
            all_test[u][item_test] = 1



        item_tensor = all_test[t.LongTensor(list(u_i_test_dict.keys()))]

        test_data = TensorDataset(t.LongTensor(users), item_tensor)    #(test_user, test_item_interactions)
        test_loader = DataLoader(test_data, batch_size=256,shuffle=True)
        user_embed, item_embed = model(uv_g,group_user)

        metric_result = {}
        for k in K:
            metric_result[k] = [[],[],[]]  #0: HR, 1: Recall, 2:ndcg
        for user, itemset in test_loader:
            predictions = t.mm(user_embed[user], t.t(item_embed))
            predictions[all_train[user].bool()] -= np.inf

            _, rating_K = t.topk(predictions, k=max(K))

            for k in K:
                result = t.gather(all_test[user],1,rating_K[:,:k])
                result_sum = t.sum(t.gather(all_test[user],1,rating_K[:,:k]),1)
                ground_truth = t.sum(itemset,1)
                hr = len(result_sum.nonzero())
                recall = t.sum(result_sum/ground_truth).item()
                # dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)

                # idcg = t.sum(t.sort(itemset,1,True)[0][:,:k]/(t.log2(t.arange(2,k+2))),1)
                dcg = t.sum(result/(t.log2(t.arange(2,k+2))),1)

                idcg = t.sum(t.sort(itemset,1,True)[0][:,:k]/(t.log2(t.arange(2,k+2))),1)
                ndcg = t.sum(dcg/idcg).item()

                metric_result[k][0].append(hr)
                metric_result[k][1].append(recall)
                metric_result[k][2].append(ndcg)
        final_result = {}
        test_user_num = len(u_i_test_dict.keys())
        for k in K:
            final_result[k] = {}
            final_result[k]['hr'] = sum(metric_result[k][0])/test_user_num
            final_result[k]['recall'] = sum(metric_result[k][1])/test_user_num
            final_result[k]['ndcg'] = sum(metric_result[k][2])/test_user_num
        return final_result
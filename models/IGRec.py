import torch
from torch._C import device
import torch.nn.functional as F
import torch.nn as nn
import pdb
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GATConv
import math

def message_func(edges):
    dic = {}
    dic['message'] = edges.src['h'] / torch.sqrt(edges.src['degree']).unsqueeze(1)
    return dic
 
def reduce_func(nodes):
    return {'h_agg': torch.sum(nodes.mailbox['message'], dim = 1) / torch.sqrt(nodes.data['degree'].unsqueeze(1))}

class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_u('h', 'm')

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg = 'm', out = 'h'), etype = etype)

            # rst = graph.dstdata['h'][dst]
            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm
            return rst

class self_gate(nn.Module):
    def __init__(self, embedding_dim):
        super(self_gate, self).__init__()
        self.embedding_dim = embedding_dim
        self.Linear = nn.Linear(self.embedding_dim, self.embedding_dim)
    
    def forward(self, embedding):
        gate = self.Linear(embedding)
        gate = torch.sigmoid(gate)
        return embedding * gate
    
class IGRec(nn.Module):
    def __init__(self,args,graph_g_u,graph,device):
        super(IGRec, self).__init__()
        #parameter 
        self.embedding_dim = args.embed_size
        self.num_users = graph.nodes('user').shape[0]
        self.num_items = graph.nodes('item').shape[0]
        self.num_aspects = args.num_aspects
        self.num_groups = graph_g_u.number_of_nodes('group')
        self.user_embedding = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.embedding_dim))
        self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.embedding_dim))
        self.attention = torch.nn.Parameter(torch.randn(1, self.embedding_dim))
        self.groupembeds = torch.nn.Parameter(torch.randn(self.num_groups, self.embedding_dim))
        # self.aspects = torch.nn.Parameter(torch.randn(self.num_users, self.num_aspects, self.embedding_dim))
        
        self.aspects = torch.nn.ModuleList()
        for i in range(args.num_aspects):
            self.aspects.append(self_gate(self.embedding_dim))
        self.softmax = nn.Softmax(dim = 0)
        # self.attention = AttentionLayer(2 * self.embedding_dim, args.drop_ratio)
        # self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)


        # self.num_heads = args.num_heads
        self.graph_g_u = dgl.edge_type_subgraph(graph_g_u,[('group', 'contain', 'user')])
        self.graph_u_g = dgl.edge_type_subgraph(graph_g_u,[('user', 'in', 'group')])
        self.device = device
        self.layer_num = args.layer_num
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                self.build_model()
        self.build_model()
        self.node_features = {'user': self.user_embedding, 'item': self.item_embedding}
        self.tau_gumbel = args.tau_gumbel
        self.isHard = args.isHard
        self.isReg = args.isReg
        self.threshold = args.threshold
        self.reg_coef = args.reg_coef

    def build_layer(self, idx):
        return LightGCNLayer()

    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            self.layers.append(h2h)


    def forward(self, graph, group_user):
        user_embed, group_emb = self.g_u_forward(group_user)
        user_embed = [user_embed]
        item_embed = [self.item_embedding]
        h = self.node_features


        for layer in self.layers:
            h_item = layer(graph, h, ('user', 'rate', 'item'))
            h_user = layer(graph, h, ('item', 'rated by', 'user'))
            h = {'user': h_user, 'item': h_item}

            user_embed.append(h_user)
            item_embed.append(h_item)
        user_embed = torch.mean(torch.stack(user_embed, dim = 0), dim = 0)
        item_embed = torch.mean(torch.stack(item_embed, dim = 0), dim = 0)

        h = {'user': user_embed, 'item': item_embed}
        return user_embed, item_embed, group_emb

    def aspect_regular(self):
        #aspect regularization
        if self.isReg:
            div_metric = None
            # N x K x dim
            for i in range(self.num_aspects):
                for j in range(i + 1, self.num_aspects):
                    sim_matrix = F.cosine_similarity(self.user_embed_asp[i], self.user_embed_asp[j])
                    mask = torch.abs(sim_matrix) > self.threshold
                    if i == 0 and j == 1:
                        div_metric = (torch.abs(torch.masked_select(sim_matrix, mask))).sum()
                    else:
                        div_metric += (torch.abs(torch.masked_select(sim_matrix, mask))).sum()

            # div_reg = self.reg_coef * div_metric
            div_reg =  div_metric
            return div_reg
        else:
            return 0

    def g_u_forward(self,group_user):
        #obtain aspects of user embeddings
        user_embed_asp = {}
        for i in range(len(self.aspects)):
            user_embed_asp[i] = self.aspects[i](self.user_embedding)
        self.user_embed_asp = user_embed_asp
        embed_centers = self.groupembeds
        #Apply softmax
        #new
        embed_member_agg = torch.zeros(self.num_groups, self.num_aspects, self.embedding_dim).to(group_user.device)

        
        for j in range(self.num_groups):
            
            asp = torch.stack([self.user_embed_asp[i][group_user[j].bool()] for i in range(self.num_aspects)],dim=0)
            H = asp * self.attention.unsqueeze(0)
            H_sum = H.sum(-1)
            weight = torch.softmax(H_sum,dim=1)
            # pdb.set_trace()
            aggregation = (weight.unsqueeze(-1) * asp).sum(1)
            embed_member_agg[j] = aggregation
        
        # embed_member_means = torch.stack([torch.mm(group_user,user_embed_asp[i])/group_user.sum(1).unsqueeze(1) for i in range(len(self.aspects))],dim=1)
        # learned_group_embed = embed_member_means.mean(1)
        
        # aspect_softmax = torch.softmax(torch.bmm(embed_member_agg, embed_centers.unsqueeze(-1)),dim=1).squeeze(-1)


        aspect_softmax = torch.bmm(embed_member_agg, embed_centers.unsqueeze(-1)).squeeze(-1)
        aspect_softmax = F.gumbel_softmax(aspect_softmax, tau=self.tau_gumbel, hard=self.isHard)
        learned_group_embed = (embed_member_agg*aspect_softmax.unsqueeze(-1)).sum(1)

        # #remove aspect
        # learned_group_embed = torch.mm(group_user,self.user_embedding)/group_user.sum(1).unsqueeze(1)

        # members_stack =torch.stack([torch.mm(group_user,user_embed_asp[i])*aspect_softmax[:,i][:,None]/group_user.sum(1).unsqueeze(1) for i in range(len(self.aspects))],dim=1)
        # learned_group_embed = members_stack.sum(1)
        updated_group_embed = (embed_centers + learned_group_embed)/2
        # Mean
        update_g_to_u = torch.mm(torch.t(group_user),updated_group_embed)/torch.t(group_user).sum(1).unsqueeze(1)
        # pdb.set_trace()
        user_embeds = (self.user_embedding + update_g_to_u)/2

        return user_embeds, updated_group_embed
        


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight


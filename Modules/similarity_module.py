import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimilarityModel(nn.Module):
    def __init__(self, emb_dim):
        super(SimilarityModel, self).__init__()
        self.emb_dim = emb_dim

    def forward(self, sub_nodes_embedding, transformed_data_embed):
        attention_net = AttentionNetwork(self.emb_dim)
        attention_net.to(device)
        # Get attention weights and weighted data_embed
        attention_weights= attention_net(sub_nodes_embedding, transformed_data_embed)
        result = []
        for i in range(len(sub_nodes_embedding)):
            temp = []
            sub = sub_nodes_embedding[i].unsqueeze(0)
            for j in range(len(transformed_data_embed)):
                temp.append(F.cosine_similarity(sub, transformed_data_embed[j].unsqueeze(0)).relu())
            result.append(temp)
        result = torch.tensor(result)
        result = result.cpu().numpy()
        result = result.flatten()
        result = torch.tensor(result)
        result = result.to(device)
        result = result * attention_weights
        return result
# 改到这，写了一个attention得到两个向量中每个点的权重
class AttentionNetwork(nn.Module):
    def __init__(self, emb_dim):
        super(AttentionNetwork, self).__init__()
        self.emb_dim = emb_dim
        self.linear_query = nn.Linear(emb_dim, emb_dim)
        self.linear_key = nn.Linear(emb_dim, emb_dim)
        self.linear_value = nn.Linear(emb_dim, 1)

    def forward(self, sub_emb, data_embed):
        query = self.linear_query(sub_emb)    # Shape: n x emb_dim
        key = self.linear_key(data_embed)     # Shape: m x emb_dim

        scores = torch.matmul(query, key.transpose(0, 1))  # Shape: n x m

        attention_weights = F.softmax(scores, dim=-1)  # Shape: n x m
        # 将二维张量转为一维
        attention_weights = attention_weights.view(-1)

        return attention_weights
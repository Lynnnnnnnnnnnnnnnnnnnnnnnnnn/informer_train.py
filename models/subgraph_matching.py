import logging
import math

import numpy as np
import torch
from numpy import sort
from Modules.embedding_module import get_embedding_module, GCN
from Modules.similarity_module import SimilarityModel
from models.time_encoding import TimeEncode
from utils.utils import get_unique_nodes, get_embedding
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Subgraph_Matching(torch.nn.Module):
  def __init__(self, neighbor_finder,nodes_embedding, last_embedding,node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.01,  embedding_module_type="graph_attention",
               n_neighbors=None,update=True
):
    super(Subgraph_Matching, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)
    self.nodes_embedding=nodes_embedding
    self.last_embedding=last_embedding
    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.update=update

    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 n_neighbors=self.n_neighbors)
  def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times])

    memory = None
    time_diffs = None


    # embedding部分
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]

    return source_node_embedding, destination_node_embedding

  def sub_embedding(self,graph):
    infeats=10
    hiddenfeats=64
    outfeats=172
    gcn=GCN(infeats,hiddenfeats,outfeats)
    gcn=gcn.to(device)
    features = torch.randn(graph.num_nodes(), infeats)
    features=features.to(device)
    embedding=gcn(graph,features)
    return embedding
  def graph_embedding(self,sources_batch,destinations_batch,timestamps_batch,edge_idxs_batch,n_degree):
    # update版本
    if self.update:
        source_node_embedding, destination_node_embedding = self.compute_temporal_embeddings \
            (sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, n_degree)
        nodes = sort(get_unique_nodes(sources_batch, destinations_batch))
        # 已用attention进行改进
        # nodes_embedding_dict, nodes_embedding = self.get_latest_embedding \
        #   (nodes, sources_batch, destinations_batch, source_node_embedding, destination_node_embedding)
        # 改进后：
        nodes_embedding = get_embedding(nodes, sources_batch, destinations_batch, source_node_embedding,
                                        destination_node_embedding)
    # 全都计算的版本
    else:
        source_node_embedding, destination_node_embedding = self.compute_temporal_embeddings\
            (sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, n_degree)

        nodes = sort(get_unique_nodes(sources_batch, destinations_batch))
        # 已用attention进行改进
        # nodes_embedding_dict, nodes_embedding = self.get_latest_embedding \
        #   (nodes, sources_batch, destinations_batch, source_node_embedding, destination_node_embedding)
        # 改进后：
        nodes_embedding = get_embedding(nodes, sources_batch, destinations_batch, source_node_embedding,
                                        destination_node_embedding)
    return nodes_embedding

  def compute_similarity(self,sub_emb, data_embed):
    result = []
    for i in range(len(sub_emb)):
        temp = []
        sub = sub_emb[i].unsqueeze(0)
        for j in range(len(data_embed)):
            temp.append(F.cosine_similarity(sub, data_embed[j].unsqueeze(0)).relu())
        result.append(temp)
    result = torch.tensor(result)
    result = result.cpu().numpy()
    result = result.flatten()
    result = torch.tensor(result)
    result = result.to(device)
    return result
  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder

  # 得到每个点最新的embedding，返回一个数组
  def get_latest_embedding(self,unique_nodes, src, des, src_embedding, des_embedding):
    nodes_embedding = {}
    i = 0
    for node1, node2 in zip(src, des):
        nodes_embedding[node1] = src_embedding[i]
        nodes_embedding[node2] = des_embedding[i]
        i += 1
    nodes_embedding = dict(sorted(nodes_embedding.items(), key=lambda item: item[0]))
    nodes_embedding = {node: nodes_embedding[node].detach().cpu().numpy() for node in unique_nodes}
    # 对字典中的值按key排序
    nodes_embedding = list(nodes_embedding.values())
    nodes_embedding = np.array(nodes_embedding)
    nodes_embedding = torch.from_numpy(nodes_embedding).to(device)
    nodes_embedding = nodes_embedding.to(torch.float32)
    return nodes_embedding

  def forward(self,sources_batch,destinations_batch,edge_idxs_batch,timestamps_batch,
              sub_g,n_degree):
      sub_nodes_embedding = self.sub_embedding(sub_g)
      nodes_embedding = self.graph_embedding(sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch,
                                             n_degree)
      emb_dim = 172

      # 创建注意力转换网络实例
      attention_informer = AttentionInformer(emb_dim)

      # 对每个嵌入矩阵进行注意力转换
      transformed_data_embed = {}
      for node, embed_matrix in nodes_embedding.items():
          transformed_matrix = attention_informer(embed_matrix)
          transformed_data_embed[node] = transformed_matrix
      # 将dict处理为tensor
      transformed_data_embed = list(transformed_data_embed.values())
      transformed_data_embed = [tensor.cpu().detach().numpy() for tensor in transformed_data_embed]
      transformed_data_embed = np.array(transformed_data_embed)
      transformed_data_embed = torch.from_numpy(transformed_data_embed).to(device)
      transformed_data_embed = transformed_data_embed.to(torch.float32)
      transformed_data_embed = transformed_data_embed.squeeze(dim=1)

      # 算到这里，下一步要写一个网络来计算两个embedding的相似度
      similarity_model = SimilarityModel(emb_dim)
      # 计算相似度
      similarity_scores = similarity_model(sub_nodes_embedding, transformed_data_embed)
      return similarity_scores
# Informer
class AttentionInformer(nn.Module):
  def __init__(self, emb_dim):
      super(AttentionInformer, self).__init__()
      self.emb_dim = emb_dim
      self.attention = nn.Linear(emb_dim, 1)  # 注意力权重
      self.linear_query = nn.Linear(emb_dim, emb_dim)
      self.linear_key = nn.Linear(emb_dim, emb_dim)
      self.attention.to(device)
      self.linear_key.to(device)
      self.linear_query.to(device)
  def forward(self, embed_matrix):
      n=embed_matrix.shape[0]
      if n <= 5:
          attention_weights = torch.softmax(self.attention(embed_matrix), dim=0)
          transformed_embed = torch.mm(attention_weights.T, embed_matrix)
          return transformed_embed
      else:
        query = self.linear_query(embed_matrix)  # Shape: n x emb_dim
        key = self.linear_key(embed_matrix)
        pick_num = math.ceil(math.log(n))
        # 随机抽取pick_num行
        indices = np.random.choice(n, pick_num, replace=False)
        # 创建新的矩阵，包含抽取的行
        new_key = key[indices, :]
        Q_K_sample = torch.matmul(query,new_key.transpose(0,1))
        # 沿着每行计算最大值
        max_values, _ = torch.max(Q_K_sample, dim=1)

        top_values, top_indices = torch.topk(max_values, pick_num)
        Q_reduced = query[top_indices]
        Q_K = torch.matmul(Q_reduced, key.transpose(0,1))
        attention_pool = AttentionPool(pick_num, n)
        attention_pool.to(device)
        attention_embed = attention_pool(Q_K,embed_matrix)
        return attention_embed
class AttentionPool(nn.Module):
    def __init__(self, pick_num, n):
        super(AttentionPool, self).__init__()
        self.pick_num = pick_num
        self.n = n
        self.attention=nn.Linear(pick_num,1)
        self.attention.to(device)
    def forward(self, Q_K,embed_matrix):
        # 使用最大池化来将其变为 1 x 32 的矩阵
        pooled_matrix, _ = torch.max(Q_K, dim=0, keepdim=True)
        attention_embed=torch.matmul(pooled_matrix,embed_matrix)
        return attention_embed
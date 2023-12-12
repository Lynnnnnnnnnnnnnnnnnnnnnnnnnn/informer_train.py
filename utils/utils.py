import os
import pickle
import random

import torch
import dgl
import networkx as nx
import numpy as np
from numpy import sort

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
      self.max_round = max_round
      self.num_round = 0

      self.epoch_count = 0
      self.best_epoch = 0

      self.last_best = None
      self.higher_better = higher_better
      self.tolerance = tolerance

  def early_stop_check(self, curr_val):
      if not self.higher_better:
        curr_val *= -1
      if self.last_best is None:
        self.last_best = curr_val
      elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
        self.last_best = curr_val
        self.num_round = 0
        self.best_epoch = self.epoch_count
      else:
        self.num_round += 1

      self.epoch_count += 1

      return self.num_round >= self.max_round
class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)
    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    source_neighbors=self.node_to_neighbors[src_idx][:i]
    source_edge_idxs=self.node_to_edge_idxs[src_idx][:i]
    source_edge_times=self.node_to_edge_timestamps[src_idx][:i]
    return source_neighbors,source_edge_idxs,source_edge_times

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times


def dynamic_subgraph_creator(sources_batch,destinations_batch,edge_idxs_batch,timestamps_batch):
  G = nx.Graph()
  for i, (src, dst, idx) in enumerate(zip(sources_batch, destinations_batch, edge_idxs_batch)):
    G.add_edge(src.item(), dst.item(), idx=idx.item())

  # 获取图中的子图
  subgraphs = list(nx.connected_components(G))

  # 找到子图中最大的点集合，并转化为np数组
  largest_subgraph_nodes = max((subgraph for subgraph in subgraphs if len(subgraph) <= 10), key=len)
  largest_subgraph_nodes = list(largest_subgraph_nodes)
  largest_subgraph_nodes = np.array(largest_subgraph_nodes)

  sub_sources_batch=[]
  sub_destinations_batch=[]
  sub_timestamps_batch=[]
  sub_edge_idxs_batch=[]

  for i, (src, dst, idx,tdx) in enumerate(zip(sources_batch, destinations_batch, edge_idxs_batch,timestamps_batch)):
      # 判断起点和终点是否在子图中
      if src in largest_subgraph_nodes and dst in largest_subgraph_nodes:
        sub_sources_batch.append(src)
        sub_destinations_batch.append(dst)
        sub_edge_idxs_batch.append(idx)
        sub_timestamps_batch.append(tdx)
  return sub_sources_batch, sub_destinations_batch,sub_timestamps_batch, sub_edge_idxs_batch

def get_unique_nodes(src,des):
  return np.unique(np.concatenate([np.unique(src),np.unique(des)]))


# 定义一个函数，输入data和step_size，输出一个包含多个graph的列表
def get_G_list(data, step_size=200):
    G_list = []
    for i in range(0, len(data.sources), step_size):
        # 获取当前的 sources, destinations, labels, timestamps, edge_idxs
        sources = data.sources[:i + step_size]
        destinations = data.destinations[:i + step_size]
        labels = data.labels[:i + step_size]
        timestamps = data.timestamps[:i + step_size]
        edge_idxs = data.edge_idxs[:i + step_size]

        # 创建 DiGraph
        G = nx.DiGraph()
        edges = zip(sources, destinations, labels, edge_idxs)
        for source, dest, label, edge_idx, timestamp in zip(sources, destinations, labels, edge_idxs, timestamps):
            G.add_node(source)
            G.add_node(dest)
            G.add_edge(source, dest, weight=label,timestamp=timestamp, edge_idx=edge_idx)
        G_list.append(G)
        # 保存G到文件
        filename = f'./data/Graph/G_{i}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(G, f)
    return G_list
def get_embedding(unique_nodes,src,des,src_embedding,des_embedding):
    nodes_embedding = {}  # 创建一个空的字典用于保存节点嵌入
    i = 0
    for node1, node2 in zip(src, des):
        if node1 not in nodes_embedding:
            nodes_embedding[node1] = []  # 初始化节点嵌入的列表
        if node2 not in nodes_embedding:
            nodes_embedding[node2] = []  # 初始化节点嵌入的列表

        nodes_embedding[node1].append(src_embedding[i])  # 将嵌入向量添加到对应节点的列表中
        nodes_embedding[node2].append(des_embedding[i])  # 将嵌入向量添加到对应节点的列表中
        i += 1
    nodes_embedding = dict(sorted(nodes_embedding.items()))

    # 将列表中的嵌入向量转换为多维数组
    for node in nodes_embedding:
        nodes_embedding[node] = torch.stack(nodes_embedding[node])
    return nodes_embedding


def get_G(start_idx):
  with open(f'data/Graph/G_{start_idx}.pkl', 'rb') as f:
    G = pickle.load(f)
  return G

def get_subgraph(G):
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    sub_G = G.subgraph(largest_cc)
    return sub_G


def get_groundtruth(start_idx,sub_g,sources_batch,destinations_batch,train,update):
    len_sub = len(sub_g.nodes())
    node=sort(get_unique_nodes(sources_batch, destinations_batch))
    len_data = len(node)
    ground_truth = np.zeros((len_sub, len_data), dtype=np.float32)
    if train is True:
        file_path=f'./data/Train_Truth/truth_{start_idx}.txt'
    else:
        file_path = f'./data/Test_Truth/truth_{start_idx}.txt'
    if os.path.getsize(file_path) == 0:
        ground_truth = ground_truth.flatten()
        ground_truth = torch.tensor(ground_truth)
        ground_truth=ground_truth.to(device)
        return ground_truth
    else:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            values = line.strip().split(':')
            for j, value in enumerate(values):
                if value:
                    data, sub = value.split(',')
                    data = int(data)
                    if update:
                        if data in node:
                            sub = int(sub)
                            indice=np.where(node==data)
                            ground_truth[sub, indice] = 1
                        else :
                            continue
        ground_truth = ground_truth.flatten()
        ground_truth = torch.tensor(ground_truth)
        ground_truth=ground_truth.to(device)
        return ground_truth


def get_value_of_sub_G(sub_G):
    src=list(node for node, _ in sub_G.in_edges())
    src=np.array(src)

    des = list(node for _, node in sub_G.out_edges())
    des = np.array(des)

    edge_index=nx.get_edge_attributes(sub_G,'edge_idx')
    edge_index=np.array(list(edge_index.values()))

    timestamps = nx.get_edge_attributes(sub_G, 'timestamp')
    timestamps=np.array(list(timestamps.values()))
    return src,des,timestamps,edge_index


def find_missing_numbers(numbers):
    # 将数字序列转换为集合，以便进行快速查找
    number_set = set(numbers)

    # 寻找缺失的数字
    missing_numbers = []
    for i in range(max(numbers)):
        if i not in number_set:
            missing_numbers.append(i)

    return missing_numbers


def create_subgraph():
    # 创建一个空的有向图
    G = dgl.DGLGraph()
    # 添加节点
    G.add_nodes(7)
    # 添加边
    # 添加边到图中
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]  # 添加的边的起始节点和目标节点索引
    src, dst = zip(*edges)
    G.add_edges(src, dst)
    G = dgl.add_self_loop(G)
    # 随机初始化节点特征
    feat_dim = 16
    G.ndata['feat'] = torch.randn(G.number_of_nodes(), feat_dim)
    G = G.to(device)
    return G

def get_neighbors(nodes, sources, destinations):
    neighbors = set()
    for node in nodes:
        for i in range(len(sources)):
            if sources[i] == node:
                neighbors.add(destinations[i])
            elif destinations[i] == node:
                neighbors.add(sources[i])
    return list(neighbors)

def evaluate_acc(result,ground_truth):
    # 将result和ground_truth转换为numpy数组
    result = np.array(result.cpu().detach().numpy())
    ground_truth = np.array(ground_truth.cpu())
    n = result.shape[0] // 3 # 计算 n 的值
    result = result.reshape(3, n)  # 将一维数组重新形状为 5 x n 的二维数组
    ground_truth = ground_truth.reshape(3, n)
    # 计算 ground_truth 中每行为 1 的数量
    num1 = np.sum(ground_truth[0, :] == 1)
    num2 = np.sum(ground_truth[1, :] == 1)
    num3 = np.sum(ground_truth[2, :] == 1)


    if num1 == num2 == num3==0:
        acc=1
        ground_truth = ground_truth.flatten()
        return acc,ground_truth
    else:
        # 获取最大值的下标
        top_indices1 = np.argpartition(result[0], -num1)[-num1:]
        top_indices2 = np.argpartition(result[1], -num2)[-num2:]
        top_indices3 = np.argpartition(result[2], -num3)[-num3:]


        result = np.zeros_like(result)
        result[0, top_indices1] = 1
        result[1, top_indices2] = 1
        result[2, top_indices3] = 10

        result = result.flatten()
        ground_truth = ground_truth.flatten()
        # 比较所有
        acc = np.sum(ground_truth == result)/ len(ground_truth)
        return acc,result
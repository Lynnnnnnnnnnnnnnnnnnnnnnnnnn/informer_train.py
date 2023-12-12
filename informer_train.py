import argparse
import math
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from utils.preprocess import data_to_Truth
from utils.utils import get_groundtruth, create_subgraph, evaluate_acc
from models.subgraph_matching import Subgraph_Matching
from utils.dataLoader import get_data
from utils.utils import EarlyStopMonitor, get_neighbor_finder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()

parser.add_argument('--train_ratio', type = float, default = 0.7,help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=200, help='Batch_size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--num_epoch', type=int, default=5, help='Number of epochs')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--n_head', type=int, default=8, help='Number of heads used in attention layer')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--update',default=True, help='Whether to use update mode')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
args = parser.parse_args()

#load data
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data,new_node_test_data =get_data('wikipedia')
# 创建sub_G
sub_g= create_subgraph()

train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)


##得到Newoutput ，注意一定要先创建文件夹
#data_to_Truth(train_data,test_data,val_data,full_data)

nodes_embedding=np.zeros((1,172),dtype=np.float32)
nodes_embedding=torch.tensor(nodes_embedding).to(device)
last_embedding={}
#初始化
subgraph_matching = Subgraph_Matching(neighbor_finder=train_ngh_finder,
            nodes_embedding=nodes_embedding,last_embedding=last_embedding, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=args.n_layer,
            n_heads=args.n_head, dropout=args.drop_out,
            embedding_module_type=args.embedding_module,
            n_neighbors=args.n_degree,update=args.update)
for param in subgraph_matching.parameters():
    param.requires_grad = True
subgraph_matching = subgraph_matching.to(device)
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(subgraph_matching.parameters(), lr=args.learning_rate)
early_stopper = EarlyStopMonitor(max_round=args.patience)



train_instance = len(train_data.sources)
train_batch = math.ceil(train_instance / args.batch_size)

test_instance = len(test_data.sources)
test_batch = math.ceil(test_instance / args.batch_size)
print(f"num_instance:{train_instance}")
print(f"train_batch:{train_batch}")
epoch_times = []
for epoch in range(args.num_epoch):
    start_epoch = time.time()
    m_loss = []
    subgraph_matching.last_embedding = {}
    batch_times=[]
    # # train
    # subgraph_matching.set_neighbor_finder(train_ngh_finder)
    # print(f"---------starting train {epoch} epoch---------")
    # subgraph_matching.train()
    # for k in range(train_batch):
    #     start_batch=time.time()
    #     print(f"starting train_batch round {k}")
    #     loss = 0
    #     batch_idx = k
    #
    #     if batch_idx >= train_batch:
    #       continue
    #
    #     start_idx = batch_idx * args.batch_size
    #     end_idx = min(train_instance, start_idx + args.batch_size)
    #
    #     if args.update:
    #         start_idx=batch_idx * args.batch_size
    #     else:
    #         start_idx=0
    #     # 服务器显存不够
    #     file_path = f'./data/Train_Truth/truth_{start_idx}.txt'
    #     if os.path.exists(file_path):
    #         pass
    #     else:
    #         break
    #     # 得到train中该batch的数据
    #     sources_batch, destinations_batch,edge_idxs_batch,timestamps_batch = \
    #         train_data.sources[start_idx:end_idx], train_data.destinations[start_idx:end_idx],\
    #         train_data.edge_idxs[start_idx: end_idx],train_data.timestamps[start_idx:end_idx]
    #
    #     # 得到ground_truth
    #     ground_truth= get_groundtruth(start_idx,sub_g,sources_batch,destinations_batch,subgraph_matching.training,args.update)
    #
    #     # 得到节点相似度
    #     result=subgraph_matching(sources_batch,destinations_batch,edge_idxs_batch,timestamps_batch,
    #                              sub_g,args.n_degree)
    #
    #     result.requires_grad_()
    #     loss = criterion(result, ground_truth)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     m_loss.append(float(loss.detach().item()))
    #     print(f"loss:{loss}")
    #     batch_time=time.time()-start_batch
    #     print(f"time:{batch_time}")
    #     batch_times.append(batch_time)
    #     torch.cuda.empty_cache()
    # torch.save(subgraph_matching.state_dict(),'./train.pkl')
    # print("result saved to train.pkl")
    # train_time = time.time() - start_epoch
    # print("train time: ", train_time)
    # with open("./time/train_time.txt", "a") as file:
    #     file.write(f"{train_time}"+"\n")

    # test
    subgraph_matching.set_neighbor_finder(full_ngh_finder)

    print(f"---------starting test {epoch} epoch---------")
    subgraph_matching.eval()
    all_acc = []
    start_test = time.time()
    for k in range(test_batch):

        # 加载之前保存的具有最佳准确率的模型参数
        subgraph_matching.load_state_dict(torch.load('./train.pkl'))

        start_batch = time.time()
        print(f"starting test_batch round {k}")
        batch_idx = k
        start_idx = batch_idx * args.batch_size
        end_idx = min(test_instance, start_idx + args.batch_size)
        # 得到test中该batch的数据
        sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch = \
            test_data.sources[0:end_idx], test_data.destinations[0:end_idx], \
            test_data.edge_idxs[0: end_idx], test_data.timestamps[0:end_idx]
        # 服务器显存不够
        file_path = f'./data/Test_Truth/truth_{start_idx}.txt'
        if os.path.exists(file_path):
            pass
        else:
            break
        # 得到ground_truth
        ground_truth = get_groundtruth(start_idx, sub_g, sources_batch, destinations_batch, subgraph_matching.training,
                                       args.update)

        # 得到节点相似度
        result = subgraph_matching(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch,
                                   sub_g, args.n_degree)

        # 得到acc
        accuracy, result = evaluate_acc(result, ground_truth)
        print(f"accuracy1 of test_batch {k}:")
        print(accuracy)
        all_acc.append(accuracy)
        with open(f"./all_acc/all_acc{epoch}.txt", "a") as file:
            file.write(str(accuracy) + "\n")
        torch.cuda.empty_cache()
    test_time = time.time() - start_test
    print("test time: ", test_time)
    with open("./time/test_time.txt", "a") as file:
        file.write(f"{test_time}" + "\n")
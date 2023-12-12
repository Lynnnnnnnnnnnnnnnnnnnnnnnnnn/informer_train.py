import csv
import math
import os
import shutil
from math import ceil

import numpy as np
import pandas as pd
from pathlib import Path

# 把feat之前的和feat分别返回
def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)



def run(data_name):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = '../data/{}.csv'.format(data_name)
  OUT_DF = '../data/ml_{}.csv'.format(data_name)
  OUT_FEAT = '../data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = '../data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  # 增加了一个全是零的第一行
  feat = np.vstack([empty, feat])

  max_idx = max(df.u.max(), df.i.max())
  # 创建一个都是0的max_idx*172的矩阵
  rand_feat = np.zeros((max_idx + 1, 172))
  # 使u和i从1开始，不是0开始
  df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)


# # 创建ml数据
# run('wikipedia')




'''---------以下为了得到ground_truth的格式转换---------'''

# 将数据转为csv格式
def data_to_csv(data,dataname):
    num_instance = len(data.sources)
    # 写入CSV文件
    with open(f'./data/{dataname}', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['', 'u', 'i', 'ts', 'label', 'idx'])  # 写入表头

        for i in range(num_instance):
            row = [i, data.sources[i], data.destinations[i],
                   data.timestamps[i], data.labels[i], data.edge_idxs[i]]
            writer.writerow(row)
    print(f"数据已成功保存到 ./data/{dataname}.csv文件中。")

# 将csv转化为txt格式
def csv_to_txt(csv_name):
    df = pd.read_csv(csv_name)

    # 提取所需的列数据
    user_id = [x for x in df['u'].to_list()]
    item_id = [x for x in df['i'].to_list()]
    state_label = df['label'].to_list()

    # 将列数据保存到txt文件
    with open(f'{csv_name}.txt', 'w') as file:
        for u, i, s in zip(user_id, item_id, state_label):
            file.write(f'{u}\t{i}\t{s}\n')

    print(f"数据已保存到{csv_name}.txt文件中。")

def split_output_file(file_path, output_dir, increment):
    # 创建保存分割文件的目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开output.txt文件进行读取
    with open(file_path, 'r') as input_file:
        lines = input_file.readlines()

        # 确定分割文件的数量
        num_files = math.ceil(len(lines) / increment)

        for i in range(num_files):
            # 确定当前分割文件的起始行和结束行
            start_idx = i * increment
            end_idx = min((i + 1) * increment, len(lines))

            # 确定当前分割文件的文件名
            file_name = f'output_{start_idx}.txt'

            # 写入指定行数的内容到输出文件
            with open(os.path.join(output_dir, file_name), 'w') as output_file:
                output_file.writelines(lines[:end_idx])

    print(f"文件已成功分割并保存到{output_dir}目录中。")

# 将txt转化为vf3可读的形式
def convert_txt_vf3(input_file, output_file):
    # 读取输入文件内容
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 解析输入文件内容
    node_attributes = []
    edges = []

    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue  # 忽略空行和以#开头的注释行

        fields = line.split('\t')

        source_node = int(fields[0])
        destination_node = int(fields[1])
        edge_attribute = float(fields[2])
        if source_node not in node_attributes:
            node_attributes.append(source_node)
        if destination_node not in node_attributes:
            node_attributes.append(destination_node)
        edges.append((source_node, destination_node, edge_attribute))
    node_attributes=sorted(node_attributes)
    # 写入新的输出文件
    with open(output_file, 'w') as file:
        # 写入节点数目
        num_nodes = node_attributes[-1]-node_attributes[0]+1
        file.write(f"{num_nodes}\n")

        # 写入节点属性
        for i in range(num_nodes):
            # 换数据集后，如果节点有标签，需要改,这里默认没有标签，所以直接写了"1"
            file.write(f"{i} {1}\n")

        # 写入边信息
        for node_id in range(num_nodes):
            node_edges = set((s, d, a) for s, d, a in edges if s == node_id)
            delete_set = set()
            for s, d, a in node_edges:
                if s == d:
                    delete_set.add((s, d, a))
            # 从 node_edges 中删除满足条件的元素
            node_edges = node_edges - delete_set

            num_edges = len(node_edges)
            file.write(f"{num_edges}\n")
            for s, d, a in node_edges:
                file.write(f"{s} {d}\n")

    print(f"数据已保存到 {output_file} 文件中。")


'''---------正式开始数据转换的函数---------
在train.py调用本函数后，将在data中得到四个文件夹，分别是full,test,train,val的newoutput
将它们四个分别进行vf3后分别得到四个Truth文件夹，将其放入./data中即可运行train
'''
def data_to_Truth(train_data,test_data,val_data,full_data):
    data_to_csv(train_data,"train_data")
    csv_to_txt('./data/train_data')
    split_output_file('./data/train_data.txt', './data/Train_Output', increment=200)
    for i in range(0,len(train_data.sources),200):
        convert_txt_vf3(f'./data/Train_Output/output_{i}.txt',f'./data/Train_newOutput/newOutput_{i}.grf')
    os.remove('./data/train_data')
    os.remove('./data/train_data.txt')
    shutil.rmtree('./data/Train_Output')

    data_to_csv(test_data,"test_data")
    csv_to_txt('./data/test_data')
    split_output_file('./data/test_data.txt', './data/Test_Output', increment=200)
    for i in range(0,len(test_data.sources),200):
        convert_txt_vf3(f'./data/Test_Output/output_{i}.txt',f'./data/Test_newOutput/newOutput_{i}.grf')
    os.remove('./data/test_data')
    os.remove('./data/test_data.txt')
    shutil.rmtree('./data/Test_Output')

    # data_to_csv(val_data,"val_data")
    # csv_to_txt('./data/val_data')
    # split_output_file('./data/val_data.txt', './data/Val_Output', increment=200)
    # for i in range(0,len(val_data.sources),200):
    #     convert_txt_vf3(f'./data/Val_Output/output_{i}.txt',f'./data/Val_newOutput/newOutput_{i}.grf')
    # os.remove('./data/val_data')
    # os.remove('./data/val_data.txt')
    # shutil.rmtree('./data/Val_Output')
    #
    # data_to_csv(full_data,"full_data")
    # csv_to_txt('./data/full_data')
    # split_output_file('./data/full_data.txt', './data/Full_Output', increment=200)
    # for i in range(134400,len(full_data.sources),200):
    #     convert_txt_vf3(f'./data/Full_Output/output_{i}.txt',f'./data/Full_newOutput/newOutput_{i}.grf')
    # os.remove('./data/full_data')
    # os.remove('./data/full_data.txt')
    # shutil.rmtree('./data/Full_Output')
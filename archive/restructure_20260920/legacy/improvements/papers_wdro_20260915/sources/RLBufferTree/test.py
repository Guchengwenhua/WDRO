import numpy as np
import torch
import os
import time
from models2.actor_critic import Actor
from utils2.rsmt_utils import *
from utils2.rsmt_utils5c import Evaluator
from utils2.log_utils import *
import argparse


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='exp', help='experiment name')
parser.add_argument('--degree', type=int, default=20, help='maximum degree of nets')
parser.add_argument('--dimension', type=int, default=3, help='terminal representation dimension')
parser.add_argument('--test_data', type=str, default='', help='test data')
parser.add_argument('--test_size', type=int, default=500, help='number of nets')
parser.add_argument('--batch_size', type=int, default=100, help='test batch size')
parser.add_argument('--capacitance_range', type=list, default=[0.1, 0.9], help='range of capacitance')
parser.add_argument('--seed', type=int, default=9, help='random seed')

args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print()
print('experiment             ', args.experiment)
print()
#net_name = 'hidden1'
#net_name = 'hidden2'
net_name = 'hidden3'
#net_name = 'ariane136'
#net_name = 'mempool_tile_wrap'
#net_name = 'NV_NVDLA_partition_p'

save_dir = 'results/' + net_name + '.txt'

all_lengths = []
all_delay = []
all_skew = []
all_time = []
all_num = []

for degree in range(10,51,10):
    model_method = 'sinModel'
    file_dir = 'saved_model/' + model_method
    model_degree = degree
    ckp_dir = file_dir + '/' + str(model_degree) + 'b.pt'
    data_dir = 'data/nets_data/' + net_name + '_degree' + str(degree) + '.txt'

    checkpoint = torch.load(ckp_dir)
    actor = Actor(args.dimension, device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    evaluator = Evaluator(degree)
    
    f = open(data_dir, "r")
    data = f.readline().split(" ")
    test_size = int(float(data[1]))
    
    x_coords = []
    cap = []
        
    for i in range(test_size):
        x = []
        y = []
        c = []
        for j in range(degree):
            data = f.readline().split(" ")
            x.append([data[0]])
            y.append([data[1]])
            c.append([data[2]])
        x = np.array(x, dtype = float) * 1e6
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = np.array(y, dtype = float) * 1e6
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        coord = np.concatenate((x, y), axis=1)
        x_coords.append(coord)
        cap.append(c)
        
    f.close()

    x_coords = np.array(x_coords, dtype = float)
    x_coords = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords))
    cap = np.array(cap, dtype = float)
    cap = (cap - np.min(cap)) / (np.max(cap) - np.min(cap))
    test_cases = np.concatenate((x_coords, cap), axis=-1)

    start_time = time.time()
    with torch.no_grad():
        outputs, _ = actor(test_cases, True)
    our_outputs = outputs.cpu().detach().numpy()     
    our_time = time.time() - start_time
    all_time.append(our_time)
    
    slack, skew, our_lengths, b_num = evaluator.eval_batch(test_cases, our_outputs, degree)
    
    all_delay.append(np.mean(slack))
    all_num.append(np.mean(b_num))
    all_skew.append(np.mean(skew))
    all_lengths.append(np.mean(our_lengths))
    
    print('finish:', degree, 'num:', test_size)
    print('time:', time.time() - start_time)
    
our_result = all_delay
our_result.extend(all_num) 
our_result.extend(all_skew) 
our_result.extend(all_lengths) 

np.savetxt(save_dir, our_result, fmt='%.8f', delimiter=' ')

print('   OK   ')
print()
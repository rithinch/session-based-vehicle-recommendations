import json
import os
import pickle
import argparse
from model import *
from azureml.core.model import Model
from utils import get_feature_vectors, pad_features

opt = argparse.Namespace()
opt.batchSize = 100
opt.hiddenSize = 100
opt.lr = 0.001
opt.lr_dc = 0.1
opt.lr_dc_step = 3
opt.l2 = 1e-5
opt.step = 1
opt.patience = 10
opt.nonhybrid = False
opt.validation = False
opt.valid_portion = 0.1
opt.model_name = 'vehicle_recommendations_model'
opt.use_features = True
opt.top_k = 20

local = True

def init():
    global model, item_to_vehicle_mappings, vehicle_to_item_mappings, item_features
    # retrieve the path to the model file using the model name

    model_path = f'outputs/{opt.model_name}.pt' if local else Model.get_model_path(opt.model_name)
    item_mapping_path = os.path.join(f'outputs/{opt.model_name}_item_veh_mapping.dat') if local else Model.get_model_path(f'item_to_veh_mappings')
    veh_mapping_path = os.path.join(f'outputs/{opt.model_name}_veh_item_mapping.dat') if local else Model.get_model_path(f'veh_to_item_mappings')
    
    item_features = pad_features(pickle.load(open(os.path.join(f'outputs/itemid_features.dat'), 'rb')))
    item_to_vehicle_mappings = pickle.load(open(item_mapping_path, 'rb')) 
    vehicle_to_item_mappings = pickle.load(open(veh_mapping_path, 'rb')) 

    n_node = len(item_to_vehicle_mappings)+1
    n_features = len(item_features[1])

    features_vector = get_feature_vectors(n_node, item_features)

    model = SessionGraph(opt, n_node, n_features, features_vector)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

def create_connection_matrix(session_sequence_list):
    inputs, mask = np.asarray([session_sequence_list]), np.asarray([[1] * len(session_sequence_list)])
    items, n_node, A, alias_inputs, features = [], [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)
    for u_input in inputs:
        node = np.unique(u_input)
        padded_nodes = node.tolist() + (max_n_node - len(node)) * [0]
        items.append(padded_nodes)
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        features.append([item_features[i] for i in padded_nodes])

    return alias_inputs, A, items, mask, features

def pre_process(raw_data):
    
    mapped_data = []
    
    for i in raw_data:
        i = i.lower()
        if i in vehicle_to_item_mappings:
            mapped_data.append(vehicle_to_item_mappings[i])

    alias_inputs, A, items, mask, features = create_connection_matrix(mapped_data)
    
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    features = trans_to_cuda(torch.Tensor(features).float())

    return items, A, mask, alias_inputs, features

def post_process(predictions):
    return [ item_to_vehicle_mappings[i] for i in predictions.topk(opt.top_k)[1].tolist()[0]]

def run(raw_data):
    
    data = raw_data if local else np.array(json.loads(raw_data)['viewed_vehicles']) #List of reg_no clicks eg [1,2,3,4]
    
    items, A, mask, alias_inputs, features = pre_process(data)

    predictions = model(items, A, mask, alias_inputs, features) # make prediction

    return post_process(predictions)


#clicks = ['GX67CUO', 'LC67RXS', 'LC67RYA', 'HX18XVH']
#init()
#print(run(clicks))
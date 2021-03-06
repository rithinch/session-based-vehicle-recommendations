import argparse
import pickle
import time
from utils import Data, split_validation, get_feature_vectors
from model import *
import os
import torch
import shutil

from azureml.core import Run

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', default='data/dataset_sample', help='dataset folder')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--top_k', type=int, default=20, help='number of recommendations to return')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', default=False, type=boolean_string, help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--model_name',  default='vehicle_recommendations_model', help='name of the model to be saved')
parser.add_argument('--output_folder',  default='outputs', help='name of the folder to save outputs')
parser.add_argument('--use_features', default=False, type=boolean_string, help='to include item features')
opt = parser.parse_args()
print(opt)


def main(run):
    train_data = pickle.load(open(os.path.join(opt.dataset_folder, 'train.dat'), 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(os.path.join(opt.dataset_folder, 'test.dat'), 'rb'))
    
    print(test_data[0][0], test_data[1][0])

    cars = pickle.load(open(os.path.join(opt.dataset_folder, 'reg_no_item_id.dat'), 'rb'))
    item_features = pickle.load(open(os.path.join(opt.dataset_folder, 'itemid_features.dat'), 'rb'))

    
    train_data = Data(train_data, shuffle=True, features=item_features)
    test_data = Data(test_data, shuffle=False, features=item_features)
    
    n_node = len(cars)+1 #1149 #6176 #5933 #unique cars
    n_feature_columns = len(item_features[1])
    features_vector = get_feature_vectors(n_node, item_features)

    run.log("Unique No. of Cars", n_node)

    model = trans_to_cuda(SessionGraph(opt, n_node, n_feature_columns=n_feature_columns, features=features_vector))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    #Before Training, Predict

    hit, mrr = predict_scores(model, test_data)
    run.log(f'Recall@{opt.top_k}', hit)
    run.log(f'MRR@{opt.top_k}', mrr)


    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        
        hit, mrr, mean_loss = train_test(model, train_data, test_data)

        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        

        #Metrics Capture
        run.log(f'Recall@{opt.top_k}', hit)
        run.log(f'MRR@{opt.top_k}', mrr)
        run.log('Mean Loss', mean_loss)

        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tMean Loss:\t%.4f,\tEpoch:\t%d,\t%d'% (hit, mrr, mean_loss, epoch, epoch))

        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

    run.log('Training Time (s)', (end - start))

    #Save Model 
    output_folder = opt.output_folder
    os.makedirs(output_folder, exist_ok=True)
    torch.save(model, f'{output_folder}/{opt.model_name}_full.pth')
    torch.save(model.state_dict(), f'{output_folder}/{opt.model_name}.pt')
    shutil.copy(os.path.join(opt.dataset_folder, 'itemid_to_vehicle_mapping.dat'), f'{output_folder}/{opt.model_name}_item_veh_mapping.dat')
    shutil.copy(os.path.join(opt.dataset_folder, 'reg_no_item_id.dat'), f'{output_folder}/{opt.model_name}_veh_item_mapping.dat')
    shutil.copy(os.path.join(opt.dataset_folder, 'itemid_features.dat'), f'{output_folder}/itemid_features.dat')

    run.log("Model Saved in Outputs", True)

if __name__ == '__main__':
    run = Run.get_context()
    main(run)

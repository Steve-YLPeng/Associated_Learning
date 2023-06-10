import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torchmetrics.functional import r2_score, auroc, f1_score
from utils import *
# from model import Model
from distributed_model import *
from distributed_model_cnn import*
from tqdm import tqdm
import os
import numpy
import gc
import time
import numpy as np
        
def get_args():

    parser = argparse.ArgumentParser('AL training')

    # model param
    parser.add_argument('--label-emb', type=int,
                        help='label embedding dimension', default=300)
    parser.add_argument('--l1-dim', type=int,
                        help='lstm1 hidden dimension', default=300)

    parser.add_argument('--dataset', type=str, default='tinyImageNet', choices=['cifar10','cifar100','tinyImageNet'])

    # training param
    parser.add_argument('--lr', type=float, help='lr', default=0.001)
    parser.add_argument('--batch-train', type=int, help='batch-size', default=128)
    parser.add_argument('--batch-test', type=int, help='batch-size', default=1024)
    parser.add_argument('--one-hot-label', type=bool,
                        help='if true then use one-hot vector as label input, else integer', default=True)
    parser.add_argument('--epoch', type=int, default=20)

    # dir param
    parser.add_argument('--save-dir', type=str, default='./ckpt/')
    parser.add_argument('--out-dir', type=str, default='./result/')

    # YLP
    parser.add_argument('--load-dir', type=str, default=None)
    parser.add_argument('--num-layer', type=int, default=4)
    parser.add_argument('--model', type=str, default='CNN_AL')
    parser.add_argument('--task', type=str, default="image")
    parser.add_argument('--feature-dim', type=int, default=0)
    parser.add_argument('--lr-schedule', type=str, default=None)
    parser.add_argument('--train-mask', type=int, default=None)
    parser.add_argument('--prefix-mask', action='store_true')
    parser.add_argument('--side-dim', type=str, default=None)
    
    # CNN
    parser.add_argument('--aug-type', type=str, default='strong')
    args = parser.parse_args()

    try:
        os.mkdir(args.save_dir)
    except:
        pass 

    return args


def test_adapt(model:alModel, data_loader:DataLoader, threshold=0.1, max_depth=None):
    model.eval()
    cor, num = 0, 0

    for x, y in data_loader:
        x, y = x.cuda(), y.cuda()
        pred, entr = model.inference_adapt(x, threshold=threshold, max_depth=max_depth)
        cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
        num += x.size(0)
    valid_acc = cor/num    
    return valid_acc

def test(model:alModel, data_loader:DataLoader, shortcut=None, task="image",):
    if task == "image":
        model.eval()
        cor, num = 0, 0
        
        for x, y in data_loader:
                               
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred = model.inference(x, shortcut)
            
            cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
            num += x.size(0)

        valid_acc = cor/num

        return valid_acc

def main():
    
    ### start of init
    
    init_start_time = time.process_time()
    args = get_args()
    if args.side_dim is not None:
        args.side_dim = [int(dim) for dim in args.side_dim.split("-")]
        args.num_layer = len(args.side_dim)
    
    
    path_name = f"{args.dataset}_{args.model}_l{args.num_layer}"
    save_path = f"{args.save_dir}/{path_name}"

    
    if args.task == "image":
        train_loader, valid_loader, class_num = get_img_data(args)
        
        if args.model == 'CNN_AL':
            model = CNN_AL(num_layer=args.num_layer, l1_dim=args.l1_dim, lr=args.lr, class_num=class_num, lab_dim=args.label_emb)
        if args.model == 'VGG_AL':
            model = VGG_AL(num_layer=args.num_layer, l1_dim=args.l1_dim, lr=args.lr, class_num=class_num, lab_dim=args.label_emb)
        if args.model == 'resnet_AL':
            model = resnet18_AL(num_layer=args.num_layer, l1_dim=args.l1_dim, lr=args.lr, class_num=class_num, lab_dim=args.label_emb)
        if args.model == 'CNN_AL_side':
            model = CNN_AL_side(num_layer=args.num_layer, l1_dim=args.l1_dim, lr=args.lr, class_num=class_num, lab_dim=args.label_emb)
        if args.model == 'VGG_AL_side':
            model = VGG_AL_side(num_layer=args.num_layer, l1_dim=args.l1_dim, lr=args.lr, class_num=class_num, lab_dim=args.label_emb)
        if args.model == 'resnet_AL_side':
            model = resnet_AL_side(num_layer=args.num_layer, l1_dim=args.l1_dim, lr=args.lr, class_num=class_num, lab_dim=args.label_emb)
            
    model = model.cuda()
    model.summary()
        
    torch.cuda.synchronize()    
    print("init_time %s"%(time.process_time()-init_start_time))
    
    test_acc_list = []
    test_time_list = []
    
    for test_count in range(10):
        train_loader, valid_loader, class_num = get_img_data(args)
        del train_loader
        
        if args.task == "image":
            
            test_acc = [[] for max_depth in range(model.num_layer)]
            test_time = [[] for max_depth in range(model.num_layer)]
            for max_depth in range(4):
  
                model.load_state_dict(torch.load(f'{save_path}_m{max_depth}.pt'))
                with torch.no_grad():
                    
                    ### shortcut testing
                    
                    for layer in range(model.num_layer):
                        gc.collect()
                        test_start_time = time.process_time()
                        acc = test(model, valid_loader, shortcut=layer+1, task=args.task)
                        
                        torch.cuda.synchronize()
                        test_time[max_depth].append(time.process_time()-test_start_time)
                        test_acc[max_depth].append(acc)
                        
                    ### adaptive testing    
                    test_threshold = [.1,.2,.3,.4,.5,.6,.7,.8,.9] 
                    for threshold in test_threshold:
                        gc.collect()
                        test_start_time = time.process_time()
                        model.data_distribution = [0 for _ in range(model.num_layer)]
                        acc = test_adapt(model, valid_loader, threshold=threshold, max_depth=max_depth+1)

                        torch.cuda.synchronize()
                        test_time[max_depth].append(time.process_time()-test_start_time)
                        test_acc[max_depth].append(acc)
                        

        test_acc_list.append(test_acc)
        test_time_list.append(test_time)
        print(test_count)
        print(test_acc)
        print(test_time)
    test_acc_avg = np.mean(test_acc_list, axis=0)
    test_acc_std = np.std(test_acc_list, axis=0)     
    test_time_avg = np.mean(test_time_list, axis=0)
    test_time_std = np.std(test_time_list, axis=0)
    print(test_acc_avg)
    print(test_acc_std)
    print(test_acc_std.shape)
    print(test_time_avg)
    print(test_time_std)
    print(test_time_std.shape)
    with open(f'{args.out_dir}/test.npy', 'wb') as f:
        np.save(f, test_acc_avg)
        np.save(f, test_acc_std)
        np.save(f, test_time_avg)
        np.save(f, test_time_std)

main()

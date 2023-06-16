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


def train(model:alModel, data_loader:DataLoader, epoch:int, aug_type:str, dataset:str, task="image", layer_mask=None):  
    if task == "image":
        
        cor, num, tot_loss = 0, 0, []
        y_out, y_tar = torch.Tensor([]),torch.Tensor([])
        data_loader = tqdm(data_loader)
        
        model.train(True, layer_mask)
        for step, (x, y) in enumerate(data_loader):
            if aug_type == "strong":
                if dataset == "cifar10" or dataset == "cifar100":
                    x = torch.cat(x)
                    y = torch.cat(y)
                else:
                    x = torch.cat(x)
                    y = torch.cat([y, y])

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            losses = model(x, y)
            
            
            """
            tot_loss.append(losses)
            #model.eval()
            with torch.no_grad():
                pred = model.inference(x)
                
                y_out = torch.cat((y_out, pred.cpu()), 0)
                y_tar = torch.cat((y_tar, y.cpu().int()), 0).int()
                cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
                num += x.size(0)
                
                data_loader.set_description(f'Train {epoch} | Acc {cor/num} ({cor}/{num})')

        train_acc = cor/num

        print(f'Train Epoch{epoch} Acc {train_acc} ({cor}/{num})')
        """

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
    out_path = f"{args.out_dir}/{path_name}"
    if args.load_dir is not None:
        load_path = f"{args.load_dir}/{path_name}"
    
        
    
    if args.task == "image":
        #train_loader, valid_loader, class_num = get_data(args)
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
            
            
    if args.load_dir != None:
        print("Load ckpt from", f'{load_path}.pt')
        
        model.load_state_dict(torch.load(f'{load_path}.pt'))
    else:
        model.apply(initialize_weights)
    model = model.cuda()
    model.summary()
        
    torch.cuda.synchronize()    
    print("init_time %s"%(time.process_time()-init_start_time))
    print('Start Training')
    total_train_time = time.process_time()
    
    ### start of training/validation
    
    if args.task == "image":
        
        
        for max_depth in range(4):
            best_AUC = 0
            best_epoch = -1
            #layer_mask = {max_depth}
            layer_mask = {*range(max_depth+1)}
            for epoch in range(int(args.epoch/model.num_layer)):
                print("gc",gc.collect())
                ep_train_start_time = time.process_time()
                train(model, train_loader, epoch, task=args.task, layer_mask=layer_mask,
                    aug_type=args.aug_type, dataset=args.dataset)
                torch.cuda.synchronize()
                print("ep%s_train_time %s"%(epoch ,time.process_time()-ep_train_start_time))
                
                with torch.no_grad():
                    
                    ### shortcut testing
                    
                    for layer in range(model.num_layer):
                    #for layer in [model.num_layer-1]:
                        ep_test_start_time = time.process_time()
                        acc = test(model, valid_loader, shortcut=layer+1, task=args.task)
                        criteria = acc
                        
                        torch.cuda.synchronize()
                        print(f'Test Epoch{epoch} layer{layer} Acc {acc}')
                        print("ep%s_l%s_test_time %s"%(epoch, layer ,time.process_time()-ep_test_start_time))
                        if layer in layer_mask:
                            if args.lr_schedule != None:
                                model.schedulerStep(layer,criteria)
                            if criteria >= best_AUC:
                                best_AUC = criteria
                                best_epoch = epoch
                                print("Save ckpt to", f'{save_path}_m{max_depth}.pt', " ,ep",epoch)
                                torch.save(model.state_dict(), f'{save_path}_m{max_depth}.pt')
                            
                    ### adaptive testing    
                    test_threshold = [.1,.2,.3,.4,.5,.6,.7,.8,.9] 
                    for threshold in test_threshold:
                        print("gc",gc.collect())
                        test_start_time = time.process_time()
                        model.data_distribution = [0 for _ in range(model.num_layer)]
                        acc = test_adapt(model, valid_loader, threshold=threshold, max_depth=max_depth+1)
                        criteria = acc
                        #if args.lr_schedule != None:
                        #    model.schedulerStep(layer,criteria)
                        torch.cuda.synchronize()
                        print(f'Test threshold {threshold} Acc {acc}')
                        print("t%s_test_time %s"%( threshold ,time.process_time()-test_start_time))
                        print("data_distribution ",model.data_distribution)
                
            print('Best AUC', best_AUC, best_epoch)

            model.load_state_dict(torch.load(f'{save_path}_m{max_depth}.pt'))

    torch.cuda.synchronize()    
    print("total_train+valid_time %s"%(time.process_time()-total_train_time))
    
main()

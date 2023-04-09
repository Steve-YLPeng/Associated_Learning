import argparse
from nltk.corpus import stopwords
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

stop_words = set(stopwords.words('english'))
        
        
def get_args():

    parser = argparse.ArgumentParser('AL training')

    # model param
    parser.add_argument('--label-emb', type=int,
                        help='label embedding dimension', default=128)
    parser.add_argument('--l1-dim', type=int,
                        help='lstm1 hidden dimension', default=300)

    parser.add_argument('--dataset', type=str, default='tinyImageNet', choices=['cifar10','cifar100','tinyImageNet'])

    # training param
    parser.add_argument('--lr', type=float, help='lr', default=0.001)
    parser.add_argument('--batch-size', type=int, help='batch-size', default=64)
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
    parser.add_argument('--same-emb', action='store_true')
    # CNN
    parser.add_argument('--aug-type', type=str, default='strong')
    args = parser.parse_args()

    try:
        os.mkdir(args.save_dir)
    except:
        pass 

    return args


def train(model:alModel, data_loader:DataLoader, epoch, aug_type, dataset, task="image", layer_mask=None):
    
        
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
            #print("size",x.shape,y.shape)
            losses = model(x, y)
            tot_loss.append(losses)
            
            #model.eval()
            pred = model.inference(x)
            
            y_out = torch.cat((y_out, pred.cpu()), 0)
            y_tar = torch.cat((y_tar, y.cpu().int()), 0).int()
            
            cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
            num += x.size(0)
            #print(f'Train {epoch} | Acc {cor/num} ({cor}/{num})')
            data_loader.set_description(f'Train {epoch} | Acc {cor/num} ({cor}/{num})')
            #gc.collect()
        train_AUC = auroc(y_out,y_tar.view(-1),num_classes=model.class_num,average='macro').item()
        train_acc = cor/num
        
        
        train_loss = numpy.sum(tot_loss, axis=0)
        model.history["train_AUC"].append(train_AUC)
        model.history["train_acc"].append(train_acc)
        model.history["train_loss"].append(train_loss)
        #print(train_loss)
        print(f'Train Epoch{epoch} Acc {train_acc} ({cor}/{num}), AUC {train_AUC}')
        del y_out
        del y_tar


def test_adapt(model:alModel, data_loader:DataLoader, threshold=0.1, max_depth=None):
    model.eval()
    cor, num = 0, 0
    y_out, y_tar, y_entr = torch.Tensor([]),torch.Tensor([]),torch.Tensor([])
    for x, y in data_loader:
        x, y = x.cuda(), y.cuda()
        pred, entr = model.inference_adapt(x, threshold=threshold, max_depth=max_depth)
        y_entr = torch.cat((y_entr, entr.cpu()), 0)            
        y_out = torch.cat((y_out, pred.cpu()), 0)
        y_tar = torch.cat((y_tar, y.cpu().int()), 0).int()
        cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
        num += x.size(0)
    valid_entr = torch.mean(y_entr).item()
    valid_AUC = auroc(y_out,y_tar.view(-1),num_classes=model.class_num,average='macro').item()
    valid_f1 = f1_score(y_out.argmax(-1).view(-1), y_tar.view(-1), average='micro')
    valid_acc = cor/num    
    return valid_AUC, valid_f1, valid_acc, valid_entr

def test(model:alModel, data_loader:DataLoader, shortcut=None, task="image",):
    if task == "image":
        model.eval()
        cor, num = 0, 0
        y_out, y_tar, y_entr = torch.Tensor([]),torch.Tensor([]),torch.Tensor([])
        
        for x, y in data_loader:
                               
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred = model.inference(x, shortcut)
            
            y_entr = torch.cat((y_entr, torch.sum(torch.special.entr(pred).cpu(),dim=-1)), 0)            
            y_out = torch.cat((y_out, pred.cpu()), 0)
            y_tar = torch.cat((y_tar, y.cpu().int()), 0).int()
            cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
            num += x.size(0)

        valid_entr = torch.mean(y_entr).item()
        valid_AUC = auroc(y_out,y_tar.view(-1),num_classes=model.class_num,average='macro').item()
        valid_f1 = f1_score(y_out.argmax(-1).view(-1), y_tar.view(-1), average='micro')
        valid_acc = cor/num

        return valid_AUC, valid_f1, valid_acc, valid_entr

def main():
    
    ### start of init
    
    init_start_time = time.process_time()
    args = get_args()
    if args.side_dim is not None:
        args.side_dim = [int(dim) for dim in args.side_dim.split("-")]
        args.num_layer = len(args.side_dim)
    
    
    path_name = f"{args.dataset}_{args.model}_l{args.num_layer}"
    
    if args.prefix_mask:
        path_name += '_prefix'
    if args.side_dim is not None:
        path_name += '_side'
    
    
    save_path = f"{args.save_dir}/{path_name}"
    out_path = f"{args.out_dir}/{path_name}"
    if args.load_dir is not None:
        load_path = f"{args.load_dir}/{path_name}"
    
        
    
    if args.task == "image":
        #train_loader, valid_loader, class_num = get_data(args)
        train_loader, valid_loader, class_num = get_img_data(args)
        if args.model == 'CNN_AL':
            model = CNN_AL(num_layer=args.num_layer, l1_dim=args.l1_dim, lr=args.lr, class_num=class_num, lab_dim=128)

            
    if args.load_dir != None:
        print("Load ckpt from", f'{load_path}.pt')
        
        model.load_state_dict(torch.load(f'{load_path}.pt'))
    else:
        model.apply(initialize_weights)
    model = model.cuda()
    model.summary()
        
    if args.train_mask != None:
        if args.prefix_mask:
            layer_mask = {*range(args.train_mask)}
        else:
            layer_mask = {args.train_mask-1}
    else:
        layer_mask = {*range(args.num_layer)}
    torch.cuda.synchronize()    
    print("init_time %s"%(time.process_time()-init_start_time))
    print('Start Training')
    total_train_time = time.process_time()
    
    ### start of training/validation
    
    if args.task == "image":
        
        best_AUC = 0
        best_epoch = -1
        for epoch in range(args.epoch):
            print("gc",gc.collect())
            ep_train_start_time = time.process_time()
            train(model, train_loader, epoch, task=args.task, layer_mask=layer_mask,
                  aug_type=args.aug_type, dataset=args.dataset)
            torch.cuda.synchronize()
            print("ep%s_train_time %s"%(epoch ,time.process_time()-ep_train_start_time))
            
            valid_acc,valid_AUC,valid_entr = [],[],[]
            with torch.no_grad():
                for layer in range(model.num_layer):
                #for layer in [model.num_layer-1]:
                    ep_test_start_time = time.process_time()
                    AUC, f1, acc, entr = test(model, valid_loader, shortcut=layer+1, task=args.task)
                    criteria = f1
                    valid_AUC.append(AUC)
                    valid_acc.append(acc)
                    valid_entr.append(entr)
                    if args.lr_schedule != None:
                        model.schedulerStep(layer,AUC)
                    torch.cuda.synchronize()
                    print(f'Test Epoch{epoch} layer{layer} Acc {acc}, AUC {AUC}, avg_entr {entr}, f1 {f1}')
                    print("ep%s_l%s_test_time %s"%(epoch, layer ,time.process_time()-ep_test_start_time))
                    if layer in layer_mask and criteria >= best_AUC:
                        best_AUC = criteria
                        best_epoch = epoch
                        print("Save ckpt to", f'{save_path}.pt', " ,ep",epoch)
                        torch.save(model.state_dict(), f'{save_path}.pt')
            model.history["valid_acc"].append(valid_acc)
            model.history["valid_AUC"].append(valid_AUC)
            model.history["valid_entr"].append(valid_entr)
            
            
        print('Best AUC', best_AUC, best_epoch)
        print('train_loss', numpy.array(model.history["train_loss"]).T.shape)
        print('valid_acc', numpy.array(model.history["valid_acc"]).T.shape)
        print('valid_AUC', numpy.array(model.history["valid_AUC"]).T.shape)
        print('train_acc', numpy.array(model.history["train_acc"]).shape)
      
    #plotResult(model, out_path, args.task)
        model.load_state_dict(torch.load(f'{save_path}.pt'))
        predicting_for_sst(args, model, vocab)
    torch.cuda.synchronize()    
    print("total_train+valid_time %s"%(time.process_time()-total_train_time))
    """
    print('Start Testing')
    if args.task == "image":
        print("Load ckpt at",f'{save_path}.pt')
        model.load_state_dict(torch.load(f'{save_path}.pt'))
        model.eval()
        with torch.no_grad():
            ### shortcut testing
            for layer in range(model.num_layer):
            #for layer in [model.num_layer-1]:
                print("gc",gc.collect())
                test_start_time = time.process_time()
                AUC, f1, acc, entr = test(model, test_loader, shortcut=layer+1, task=args.task)
                torch.cuda.synchronize()
                print(f'Test layer{layer} Acc {acc}, AUC {AUC}, avg_entr {entr}, f1 {f1}')
                print("l%s_test_time %s"%(layer ,time.process_time()-test_start_time))
            
            ### adaptive testing    
            test_threshold = [.1,.2,.3,.4,.5,.6,.7,.8,.9] 
            for threshold in test_threshold:
                print("gc",gc.collect())
                test_start_time = time.process_time()
                AUC,f1, acc, entr = test_adapt(model, test_loader, threshold=threshold, max_depth=args.train_mask)
                torch.cuda.synchronize()
                print(f'Test threshold {threshold} Acc {acc}, AUC {AUC}, avg_entr {entr}, f1 {f1}')
                print("t%s_test_time %s"%( threshold ,time.process_time()-test_start_time))
    """    
main()

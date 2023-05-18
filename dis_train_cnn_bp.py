import argparse
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, f1_score
from utils import *
# from model import Model
from distributed_model import *
from distributed_model_cnn import*
from tqdm import tqdm
import os
import numpy
import gc
import time

def adjust_learning_rate(optimizer, base_lr, end_lr, step, max_steps):
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    lr = base_lr * q + end_lr * (1 - q)
    set_lr(optimizer, lr)
    return lr

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
                
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


def train(model:nn.Module, data_loader:DataLoader, epoch:int, aug_type:str, dataset:str, optimizer, task="image",):  
    if task == "image":
        
        cor, num, tot_loss = 0, 0, []
        y_out, y_tar = torch.Tensor([]),torch.Tensor([])
        data_loader = tqdm(data_loader)
        
        
        for step, (x, y) in enumerate(data_loader):
            if aug_type == "strong":
                if dataset == "cifar10" or dataset == "cifar100":
                    x = torch.cat(x)
                    y = torch.cat(y)
                else:
                    x = torch.cat(x)
                    y = torch.cat([y, y])

            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            
            model.train()
            optimizer.zero_grad()
            losses = model(x, y)
            losses.backward()
            optimizer.step()
            
            
            
            model.eval()
            with torch.no_grad():
                pred = model(x, y)
                
                y_out = torch.cat((y_out, pred.cpu()), 0)
                y_tar = torch.cat((y_tar, y.cpu().int()), 0).int()
                cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
                num += x.size(0)
                
                data_loader.set_description(f'Train {epoch} | Acc {cor/num} ({cor}/{num})')

        train_AUC = auroc(y_out,y_tar.view(-1),num_classes=model.class_num,average='macro').item()
        train_acc = cor/num
        
        

        #print(train_loss)
        print(f'Train Epoch{epoch} Acc {train_acc} ({cor}/{num}), AUC {train_AUC}, loss {losses.item()}')
        del y_out
        del y_tar


def test(model:alModel, data_loader:DataLoader, task="image",):
    if task == "image":
        model.eval()
        cor, num = 0, 0
        y_out, y_tar, y_entr = torch.Tensor([]),torch.Tensor([]),torch.Tensor([])
        
        for x, y in data_loader:
                               
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred = model(x, y)
            
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
    
    path_name = f"{args.dataset}_{args.model}_l{args.num_layer}"
    
    save_path = f"{args.save_dir}/{path_name}"
    out_path = f"{args.out_dir}/{path_name}"
    if args.load_dir is not None:
        load_path = f"{args.load_dir}/{path_name}"
    
        
    
    if args.task == "image":
        #train_loader, valid_loader, class_num = get_data(args)
        train_loader, valid_loader, class_num = get_img_data(args)
        if args.model == 'CNN':
            model = CNN(class_num=class_num)  
        if args.model == 'VGG':
            model = VGG(class_num=class_num)  
        if args.model == 'resnet':
            model = resnet18(class_num=class_num)   
            
    if args.load_dir != None:
        print("Load ckpt from", f'{load_path}.pt')
        
        model.load_state_dict(torch.load(f'{load_path}.pt'))
    else:
        model.apply(initialize_weights)
    model = model.cuda()
    model.summary()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
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
            
            
            train(model, train_loader, epoch, task=args.task, optimizer=optimizer,
                  aug_type=args.aug_type, dataset=args.dataset)
            torch.cuda.synchronize()
            print("ep%s_train_time %s"%(epoch ,time.process_time()-ep_train_start_time))
            
            valid_acc,valid_AUC,valid_entr = [],[],[]
            with torch.no_grad():
                
                ### testing
                ep_test_start_time = time.process_time()
                AUC, f1, acc, entr = test(model, valid_loader, task=args.task)
                criteria = f1
                valid_AUC.append(AUC)
                valid_acc.append(acc)
                valid_entr.append(entr)

                torch.cuda.synchronize()
                print(f'Test Epoch{epoch} Acc {acc}, AUC {AUC}, avg_entr {entr}, f1 {f1}')
                print("ep%s_test_time %s"%(epoch, time.process_time()-ep_test_start_time))
                if criteria >= best_AUC:
                    best_AUC = criteria
                    best_epoch = epoch
                    print("Save ckpt to", f'{save_path}.pt', " ,ep",epoch)
                    torch.save(model.state_dict(), f'{save_path}.pt')


            
            
        print('Best AUC', best_AUC, best_epoch)

      
    #plotResult(model, out_path, args.task)

    torch.cuda.synchronize()    
    print("total_train+valid_time %s"%(time.process_time()-total_train_time))

main()

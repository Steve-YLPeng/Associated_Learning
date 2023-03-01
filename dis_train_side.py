import argparse
from nltk.corpus import stopwords
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torchmetrics.functional import r2_score, auroc
from utils import *
# from model import Model
from distributed_model import *
from tqdm import tqdm
import os
import numpy
import gc
import time

stop_words = set(stopwords.words('english'))
        
        
def get_args():

    parser = argparse.ArgumentParser('AL training')

    # model param
    parser.add_argument('--emb-dim', type=int,
                        help='word embedding dimension', default=300)
    parser.add_argument('--label-emb', type=int,
                        help='label embedding dimension', default=128)
    parser.add_argument('--l1-dim', type=int,
                        help='lstm1 hidden dimension', default=300)

    parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)
    parser.add_argument('--max-len', type=int, help='max input length', default=128)
    parser.add_argument('--dataset', type=str, default='ag_news', choices=['ag_news', 'dbpedia_14', 'banking77', 'emotion', 'rotten_tomatoes','imdb', 'clinc_oos', 'yelp_review_full', 'sst2', 
                                                                           'paint','ailerons',"criteo","ca_housing","kdd99"])
    parser.add_argument('--word-vec', type=str, default='glove')

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
    parser.add_argument('--num-layer', type=int, default=3)
    parser.add_argument('--model', type=str, default='lstmal')
    parser.add_argument('--task', type=str, default="text")
    parser.add_argument('--feature-dim', type=int, default=0)
    parser.add_argument('--lr-schedule', type=str, default=None)
    parser.add_argument('--train-mask', type=int, default=None)
    parser.add_argument('--prefix-mask', action='store_true')
    parser.add_argument('--side-dim', type=str, default=None)
    parser.add_argument('--same-emb', action='store_true')

    args = parser.parse_args()

    try:
        os.mkdir(args.save_dir)
    except:
        pass 

    return args


def train(model:alModel, data_loader:DataLoader, epoch, task="text", layer_mask=None):
    
        
    if task == "text" or task == "classification":
        
        cor, num, tot_loss = 0, 0, []
        y_out, y_tar = torch.Tensor([]),torch.Tensor([])
        data_loader = tqdm(data_loader)
        
        model.train(True, layer_mask)
        for step, (x, y) in enumerate(data_loader):
            
            #print(step)
            x, y = x.cuda(), y.cuda()
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
        #print("train_gc",gc.collect())
        
    elif task == "regression":
        out_loss, num, tot_loss = 0, 0, []
        y_out, y_tar = torch.Tensor([]),torch.Tensor([])
        data_loader = tqdm(data_loader)
        for step, (x, y) in enumerate(data_loader):
            #print(x)
            #print(y)
            
            x, y = x.cuda(), y.cuda()
            losses = model(x, y)
            tot_loss.append(losses)
                
            pred = model.inference(x)
            y_out = torch.cat((y_out, pred.cpu()), 0)
            y_tar = torch.cat((y_tar, y.cpu()), 0)
            #cor += (pred.argmax(-1) == y).sum().item()
            out_loss += mse_loss(pred, y, reduction='sum')
            num += x.size(0)
            
            data_loader.set_description(f'Train {epoch} | out_loss {torch.sqrt(out_loss/num)}')
            #gc.collect()
            
        #train_acc = cor/num
        #train_out = torch.sqrt(out_loss/num).item()
        train_out = mse_loss(y_out, y_tar).item()
        train_r2 = r2_score(y_out, y_tar).item()
        train_loss = numpy.sum(tot_loss, axis=0)
        model.history["train_out"].append(train_out)
        model.history["train_loss"].append(train_loss)
        model.history["train_r2"].append(train_r2)
        #print(train_loss)
        #loss = torch.sqrt(mse_loss(y_out, y_tar))
        #del y_out
        #del y_tar
        #print("train_gc",gc.collect())
        print(f'Train Epoch{epoch} out_loss {train_out}, R2 {train_r2}')

def test(model:alModel, data_loader:DataLoader, shortcut=None, task="text"):
    if task == "text" or task == "classification":
        model.eval()
        cor, num = 0, 0
        #y_out, y_tar = torch.Tensor([]),torch.Tensor([])
        y_out, y_tar, y_entr = torch.Tensor([]),torch.Tensor([]),torch.Tensor([])
        
        #data_loader = tqdm(data_loader)
        for x, y in data_loader:
            x, y = x.cuda(), y.cuda()
            pred = model.inference(x, shortcut)
            
            y_entr = torch.cat((y_entr, torch.sum(torch.special.entr(pred).cpu(),dim=-1)), 0)            
            y_out = torch.cat((y_out, pred.cpu()), 0)
            y_tar = torch.cat((y_tar, y.cpu().int()), 0).int()
            #cor += (pred.argmax(-1) == y).sum().item()
            cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
            num += x.size(0)
            #gc.collect()
        #print(y_entr)
        valid_entr = torch.mean(y_entr).item()
        valid_AUC = auroc(y_out,y_tar.view(-1),num_classes=model.class_num,average='macro').item()
        valid_acc = cor/num
        #del y_out
        #del y_tar
        #print("test_gc",gc.collect())
        return valid_AUC, valid_acc, valid_entr
    
    elif task == "regression":
        model.eval()
        out_loss, num, tot_loss = 0, 0, []
        y_out, y_tar = torch.Tensor([]),torch.Tensor([])
        #data_loader = tqdm(data_loader)
        for x, y in data_loader:
            x, y = x.cuda(), y.cuda()
            pred = model.inference(x, shortcut)
            y_out = torch.cat((y_out, pred.cpu()), 0)
            y_tar = torch.cat((y_tar, y.cpu()), 0)
            out_loss += mse_loss(pred, y, reduction='sum')
            num += x.size(0)
            #gc.collect()
            
        valid_out = mse_loss(y_out, y_tar).item()
        valid_r2 = r2_score(y_out, y_tar).item()
        #return torch.sqrt(out_loss/num).item()
        #del y_out
        #del y_tar
        return valid_out, valid_r2

def predicting_for_sst(args, model, vocab):

    test_data = load_dataset('sst2', split='test')
    test_text = [b['sentence'] for b in test_data]
    test_label = [b['label'] for b in test_data]
    clean_test = [data_preprocessing(t, True) for t in test_text]
    
    testset = Textset(clean_test, test_label, vocab, args.max_len)
    valid_loader = DataLoader(testset, batch_size=1, collate_fn = testset.collate)

    all_pred = []
    all_idx = []
    for i, (x, y) in enumerate(valid_loader):
        x = x.cuda()
        pred = model.inference(x).argmax(1).squeeze(0)
        all_pred.append(pred.item())
        all_idx.append(i)
    
    pred_file = {'index':all_idx, 'prediction':all_pred}
    output = pd.DataFrame(pred_file)
    output.to_csv('SST-2.tsv', sep='\t', index=False)

def main():
    init_start_time = time.time()
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
    
        
    
    if args.task == "text":
        train_loader, valid_loader, test_loader, class_num, vocab = get_data(args)
        word_vec = get_word_vector(vocab, args.word_vec)
        
        if args.model == 'lstmal':
            model = LSTMModelML(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
        elif args.model == 'linearal':
            model = LinearModelML(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
        elif args.model == 'transformeral':
            model = TransformerModelML(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
        
        elif args.model == 'transformeralside':
            model = TransformerALsideText(vocab_size=len(vocab), num_layer=args.num_layer, side_dim=args.side_dim, same_emb=args.same_emb,
                                          emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
        elif args.model == 'linearalside':
            model = LinearALsideText(vocab_size=len(vocab), num_layer=args.num_layer, side_dim=args.side_dim, same_emb=args.same_emb,
                                          emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
        elif args.model == 'lstmalside':
            model = LSTMALsideText(vocab_size=len(vocab), num_layer=args.num_layer, side_dim=args.side_dim, same_emb=args.same_emb,
                                          emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
        
        
    elif args.task == "classification":
        train_loader, valid_loader, test_loader, class_num  = get_data(args)
        if args.model == 'linearal':
            model = LinearALCLS(num_layer=args.num_layer, feature_dim=args.feature_dim, class_num=class_num, l1_dim=args.l1_dim, lab_dim=args.label_emb, lr=args.lr)
        elif args.model == 'linearalside' :
            model = LinearALsideCLS(num_layer=args.num_layer, side_dim=args.side_dim, class_num=class_num, l1_dim=args.l1_dim, lab_dim=args.label_emb, lr=args.lr)
    elif args.task == "regression":
        train_loader, valid_loader, test_loader, target_num  = get_data(args)
        if args.model == 'linearal':
            model = LinearALRegress(num_layer=args.num_layer, feature_dim=args.feature_dim, class_num=1, l1_dim=args.l1_dim, lab_dim=args.label_emb, lr=args.lr)

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
        
    print("init_time %s"%(time.time()-init_start_time))
    print('Start Training')
    total_train_time = time.time()
    
    if args.task == "text" or args.task == "classification":
        
        best_AUC = 0
        for epoch in range(args.epoch):
            print("gc",gc.collect())
            ep_train_start_time = time.time()
            train(model, train_loader, epoch, task=args.task, layer_mask=layer_mask)
            print("ep%s_train_time %s"%(epoch ,time.time()-ep_train_start_time))
            
            valid_acc,valid_AUC,valid_entr = [],[],[]
            with torch.no_grad():
                for layer in range(model.num_layer):
                    ep_test_start_time = time.time()
                    AUC, acc, entr = test(model, valid_loader, shortcut=layer+1, task=args.task)
                    valid_AUC.append(AUC)
                    valid_acc.append(acc)
                    valid_entr.append(entr)
                    if args.lr_schedule != None:
                        model.schedulerStep(layer,AUC)
                    print(f'Test Epoch{epoch} layer{layer} Acc {acc}, AUC {AUC}, avg_entr {entr}')
                    print("ep%s_l%s_test_time %s"%(epoch, layer ,time.time()-ep_test_start_time))
                    if layer in layer_mask and AUC >= best_AUC:
                        best_AUC = AUC
                        print("Save ckpt to", f'{save_path}.pt', " ,ep",epoch)
                        torch.save(model.state_dict(), f'{save_path}.pt')
            model.history["valid_acc"].append(valid_acc)
            model.history["valid_AUC"].append(valid_AUC)
            model.history["valid_entr"].append(valid_entr)
            
            
        print('Best AUC', best_AUC)
        print('train_loss', numpy.array(model.history["train_loss"]).T.shape)
        print('valid_acc', numpy.array(model.history["valid_acc"]).T.shape)
        print('valid_AUC', numpy.array(model.history["valid_AUC"]).T.shape)
        print('train_acc', numpy.array(model.history["train_acc"]).shape)
        
    elif args.task == "regression":
        best_r2, best_layer = 0,-1
        for epoch in range(args.epoch):
            print("gc",gc.collect())
            train(model, train_loader, epoch, task="regression", layer_mask=layer_mask)
            valid_out, valid_r2 = [],[]
            
            with torch.no_grad():
                for layer in range(model.num_layer):
                    loss, r2 = test(model, valid_loader, shortcut=layer+1, task="regression")
                    valid_out.append(loss)
                    valid_r2.append(r2)
                    print(f'Test Epoch{epoch} layer{layer} out_loss {loss}, R2 {r2}')
                    if args.lr_schedule != None:
                        model.schedulerStep(layer,r2)
                    
                    if layer in layer_mask and r2 > best_r2:
                        best_r2 = r2
                        best_layer = layer
                        torch.save(model.state_dict(), f'{save_path}.pt')
                    
            model.history["valid_out"].append(valid_out)
            model.history["valid_r2"].append(valid_r2)
            

        print(f'Best r2 {best_r2} at L{best_layer}' )
            
    plotResult(model, out_path, args.task)
    
    if args.dataset == 'sst2':
        model.load_state_dict(torch.load(f'{save_path}.pt'))
        predicting_for_sst(args, model, vocab)
        
    print("total_train+valid_time %s"%(time.time()-total_train_time))
    print('Start Testing')
    if args.task == "text" or args.task == "classification":
        print("Load ckpt at",f'{save_path}.pt')
        model.load_state_dict(torch.load(f'{save_path}.pt'))
        model.eval()
        with torch.no_grad():
            for layer in range(model.num_layer):
                test_start_time = time.time()
                AUC, acc, entr = test(model, test_loader, shortcut=layer+1, task=args.task)
                print(f'Test layer{layer} Acc {acc}, AUC {AUC}, avg_entr {entr}')
                print("l%s_test_time %s"%(layer ,time.time()-test_start_time))
                
                
                """
                ep_test_start_time = time.time()
                y_out, y_tar, y_entr = torch.Tensor([]),torch.Tensor([]),torch.Tensor([])
                cor,num = 0,0
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    pred = model.inference(x, layer+1)
                    
                    y_entr = torch.cat((y_entr, torch.sum(torch.special.entr(pred).cpu(),dim=-1)), 0)
                    #print(torch.special.entr(pred))
                    y_out = torch.cat((y_out, pred.argmax(-1).view(-1).cpu().int()), 0).int()
                    y_tar = torch.cat((y_tar, y.view(-1).cpu().int()), 0).int()
                    #gc.collect()
                    cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
                    num += x.size(0)

                test_entr = torch.mean(y_entr).item()
                test_AUC = auroc(y_out,y_tar.view(-1),num_classes=model.class_num,average='macro').item()
                test_acc = cor/num
                print(f'Test l{layer} Acc {test_acc}, AUC {test_AUC}, avg_entr {test_entr}')
                print("l%s_test_time %s"%( layer ,time.time()-ep_test_start_time))
                print(y_entr)
                plotConfusionMatrix(y_out, y_tar, [str(i) for i in range(model.class_num)], f"{out_path}_test_l{layer}")
                """
                
        
    
main()

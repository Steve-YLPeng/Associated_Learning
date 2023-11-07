import os
import time
import argparse
import torch
from tqdm import tqdm
from utils import *
from distributed_model import *


# Function to parse arguments 
def get_args():
    parser = argparse.ArgumentParser('AL training')

    # model arguments 
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

    # training arguments 
    parser.add_argument('--lr', type=float, help='lr', default=0.001)
    parser.add_argument('--batch-train', type=int, help='batch-size', default=128)
    parser.add_argument('--batch-test', type=int, help='batch-size', default=1024)
    parser.add_argument('--one-hot-label', type=bool,
                        help='if true then use one-hot vector as label input, else integer', default=True)
    parser.add_argument('--epoch', type=int, default=20)

    # dir arguments 
    parser.add_argument('--save-dir', type=str, default='./ckpt/')
    parser.add_argument('--out-dir', type=str, default='./result/')

    # YLP: multi-layer arguments 
    parser.add_argument('--load-dir', type=str, default=None)
    parser.add_argument('--num-layer', type=int, default=3)
    parser.add_argument('--model', type=str, default='lstmal')
    parser.add_argument('--task', type=str, default="text")
    parser.add_argument('--feature-dim', type=int, default=0)
    parser.add_argument('--lr-schedule', type=str, default="plateau")
    parser.add_argument('--train-mask', type=int, default=None)
    parser.add_argument('--prefix-mask', action='store_true')
    parser.add_argument('--side-dim', type=str, default=None)
    parser.add_argument('--same-emb', action='store_true')

    args = parser.parse_args()
    if args.side_dim is not None:
        args.side_dim = [int(dim) for dim in args.side_dim.split("-")]
        args.num_layer = len(args.side_dim)

    return args


# Function to get the layer mask
def get_mask(args):
    if args.train_mask != None:
        if args.prefix_mask:
            layer_mask = {*range(args.train_mask)}
        else:
            layer_mask = {args.train_mask-1}
    else:
        layer_mask = {*range(args.num_layer)}
    return layer_mask


# Function to initialize the RNN model (Linear, LSTM, Transformer) based on training method (multi-layer AL, sideInput)
def initialize_model(args, vocab, word_vec, class_num:int):
    if args.model == 'lstmal':
        model = LSTM_AL(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
    elif args.model == 'linearal':
        model = Linear_AL(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
    elif args.model == 'transformeral':
        model = Transformer_AL(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
    
    elif args.model == 'transformeralside':
        model = Transformer_AL_Side(vocab_size=len(vocab), num_layer=args.num_layer, side_dim=args.side_dim, same_emb=args.same_emb,
                                        emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
    elif args.model == 'linearalside':
        model = Linear_AL_Side(vocab_size=len(vocab), num_layer=args.num_layer, side_dim=args.side_dim, same_emb=args.same_emb,
                                        emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
    elif args.model == 'lstmalside':
        model = LSTM_AL_Side(vocab_size=len(vocab), num_layer=args.num_layer, side_dim=args.side_dim, same_emb=args.same_emb,
                                        emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
    
    return model


# Function for training model 1 epoch
def train(model:ALModel_Template, data_loader, current_epoch:int, task:str="text", layer_mask=None):
    if task == "text":
        cor, num, tot_loss = 0, 0, []
        y_out, y_tar = torch.Tensor([]),torch.Tensor([])
        data_loader = tqdm(data_loader)
        
        model.train(True, layer_mask)

        for (x, y) in data_loader:
            # training step
            x, y = x.cuda(), y.cuda()
            losses = model(x, y)
            tot_loss.append(losses)

            # validation step
            with torch.no_grad():
                pred = model.inference(x)
                y_out = torch.cat((y_out, pred.cpu()), 0)
                y_tar = torch.cat((y_tar, y.cpu().int()), 0).int()
                cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
                num += x.size(0)
                
                data_loader.set_description(f'Train {current_epoch} | Acc {cor/num} ({cor}/{num})')
        train_acc = cor/num
        
        print(f'Train Epoch{current_epoch} Acc {train_acc} ({cor}/{num})')


# Function for adaptive testing 1 epoch
def test_adapt(model:ALModel_Template, data_loader, threshold=0.1, max_depth=None):
    model.eval()
    cor, num = 0, 0
    for x, y in data_loader:
        x, y = x.cuda(), y.cuda()
        pred, entr = model.inference_adapt(x, threshold=threshold, max_depth=max_depth)
        cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
        num += x.size(0)

    valid_acc = cor/num    
    return valid_acc


# Function for regular testing (shortcut) 1 epoch
def test(model:ALModel_Template, data_loader, shortcut=None, task="text"):
    if task == "text":
        model.eval()
        cor, num = 0, 0
        
        for x, y in data_loader:
            x, y = x.cuda(), y.cuda()
            pred = model.inference(x, shortcut)
            cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
            num += x.size(0)

        valid_acc = cor/num
        return valid_acc
    

def main():
    # Time tracking for initialization
    init_start_time = time.process_time()

    # Initialize arguments and path settings
    args = get_args()
    os.makedirs(args.save_dir)
    path_name = f"{args.dataset}_{args.model}_l{args.num_layer}"
    save_path = f"{args.save_dir}/{path_name}"
    out_path = f"{args.out_dir}/{path_name}"
    if args.load_dir is not None:
        load_path = f"{args.load_dir}/{path_name}"
    
    # Load necessary data
    train_loader, valid_loader, class_num, vocab = get_data(args)
    layer_mask = get_mask(args)
    word_vec = get_word_vector(vocab, args.word_vec)    

    # Initialize the model
    model = initialize_model(args, vocab, word_vec, class_num, path_name)

    if args.load_dir is not None:
        # Load model from the ckpt
        model.load_state_dict(torch.load(f'{load_path}.pt'))
    else:
        model.apply(initialize_weights)
    model = model.cuda()
    model.summary()
    
    torch.cuda.synchronize()    
    print("init_time %s"%(time.process_time()-init_start_time))
    print('Start Training')
    total_train_time = time.process_time()
    
    # Training and validation loop      
    best_acc = 0
    best_para = -1
    best_epoch = -1
    for epoch in range(args.epoch):
        # Time tracking for training
        ep_train_start_time = time.process_time()
        
        # Training step
        train(model, train_loader, epoch, task=args.task, layer_mask=layer_mask)
        
        torch.cuda.synchronize()
        print("ep%s_train_time %s"%(epoch ,time.process_time()-ep_train_start_time))

        # Validation steps for shortcuts and adaptive testing        
        with torch.no_grad():
            # shortcut testing step
            for layer in range(model.num_layer):
                ep_test_start_time = time.process_time()
                acc = test(model, valid_loader, shortcut=layer+1, task=args.task)
                criteria = acc
                if args.lr_schedule != None:
                    model.schedulerStep(layer,criteria)
                torch.cuda.synchronize()
                print(f'Test Epoch{epoch} layer{layer} Acc {acc}')
                print("ep%s_l%s_test_time %s"%(epoch, layer ,time.process_time()-ep_test_start_time))
                if layer in layer_mask and criteria >= best_acc:
                    best_acc = criteria
                    best_para = layer
                    best_epoch = epoch
                    print("Save ckpt to", f'{save_path}.pt', " ,ep",epoch)
                    torch.save(model.state_dict(), f'{save_path}.pt')

            # adaptive testing step 
            test_threshold = [.1,.2,.3,.4,.5,.6,.7,.8,.9] 
            for threshold in test_threshold:
                test_start_time = time.process_time()
                model.data_distribution = [0 for _ in range(model.num_layer)]
                acc = test_adapt(model, valid_loader, threshold=threshold, max_depth=args.train_mask)
                criteria = acc
                torch.cuda.synchronize()
                print(f'Test threshold {threshold} Acc {acc}')
                print("t%s_test_time %s"%( threshold ,time.process_time()-test_start_time))
                print("data_distribution ",model.data_distribution)
                if criteria >= best_acc:
                    best_acc = criteria
                    best_epoch = epoch
                    print("Save ckpt to", f'{save_path}.pt', " ,ep",epoch)
                    torch.save(model.state_dict(), f'{save_path}.pt')

    print('Best Acc', best_acc, best_epoch, best_para)
    
    torch.cuda.synchronize()    
    print("total_train+valid_time %s"%(time.process_time()-total_train_time))

if __name__ == '__main__':
    main()

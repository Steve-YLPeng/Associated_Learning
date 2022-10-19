import argparse
from nltk.corpus import stopwords
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torchmetrics.functional import r2_score
from utils import *
# from model import Model
from distributed_model import LSTMModelML, LinearModelML, TransformerModelML, LinearALRegress, LinearALCLS, LinearALsideCLS
from tqdm import tqdm
import os
import numpy
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))

def plotConfusionMatrix(y_pred, y_true, label_name, save_filename=''):
    from sklearn.metrics import confusion_matrix    
    import seaborn as sn
    import pandas as pd

    labels = label_name
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    df_cm = pd.DataFrame(cm, range(len(cm)), range(len(cm)))
    # plt.figure(figsize=(10,7))
    #sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True,vmin=0, vmax=1000) # font size
    plt.savefig(save_filename+"_cm.png")
    plt.show()
    return
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(save_filename)
    fig.colorbar(cax,)
    #ax.set_xticklabels([''] + labels)
    #ax.set_yticklabels([''] + labels)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.savefig(save_filename+"_cm.png")
    #plt.show()
    
def plotResult(model, save_filename, task):
    
    history = model.history
    # plot epoch loss
    epoch_layers_loss = numpy.array(history["train_loss"]).T
    epoch_layers_ae_loss = epoch_layers_loss[0,:,:]
    epoch_layers_as_loss = epoch_layers_loss[1,:,:]
    for idx,loss in enumerate(epoch_layers_ae_loss):
        plt.plot(loss, label='loss L'+str(idx))
    plt.legend()
    plt.savefig(save_filename+"_ae_loss.png")
    plt.show()
    for idx,loss in enumerate(epoch_layers_as_loss):
        plt.plot(loss, label='loss L'+str(idx))
    plt.legend()
    plt.savefig(save_filename+"_as_loss.png")
    plt.show()
    if task == "text" or task == "classification":
        # plot epoch acc
        epoch_valid_acc = numpy.array(history["valid_acc"]).T
        for idx,acc in enumerate(epoch_valid_acc):
            plt.plot(acc, label='valid acc L'+str(idx+1))
        plt.plot(history["train_acc"], "k", label='train_acc' )
        #plt.ylim(0.9, 1.0)
        plt.legend()
        plt.savefig(save_filename+"_acc.png")
        plt.show()
    if task == "regression":
        epoch_valid_out = numpy.array(history["valid_out"]).T
        for idx,acc in enumerate(epoch_valid_out):
            plt.plot(acc, label='valid out_loss L'+str(idx+1))
        plt.plot(history["train_out"], "k", label='train_out' )
        plt.legend()
        plt.savefig(save_filename+"_out.png")
        plt.show()
        
        epoch_valid_r2 = numpy.array(history["valid_r2"]).T
        for idx,acc in enumerate(epoch_valid_r2):
            plt.plot(acc, label='valid_r2 L'+str(idx+1))
        
        plt.plot(history["train_r2"], "k", label='train_r2' )
        plt.legend()
        plt.savefig(save_filename+"_r2.png")
        plt.show()
        
        
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
    parser.add_argument('--max-len', type=int, help='max input length', default=200)
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

    # YLP
    parser.add_argument('--num-layer', type=int, default=3)
    parser.add_argument('--model', type=str, default='lstmal')
    parser.add_argument('--task', type=str, default="text")
    parser.add_argument('--feature-dim', type=int, default=0)
    
    args = parser.parse_args()

    try:
        os.mkdir(args.save_dir)
    except:
        pass 

    return args

def get_data(args):

    if args.dataset == 'imdb':
        import pandas as pd
        from sklearn.model_selection import train_test_split
        class_num = 2
        df = pd.read_csv('./IMDB_Dataset.csv')
        df['cleaned_reviews'] = df['review'].apply(data_preprocessing, True)
        corpus = [word for text in df['cleaned_reviews'] for word in text.split()]
        text = [t for t in df['cleaned_reviews']]
        label = []
        for t in df['sentiment']:
            if t == 'negative':
                label.append(1)
            else:
                label.append(0)
        vocab = create_vocab(corpus)
        clean_train, clean_test, train_label, test_label = train_test_split(text, label, test_size=0.2)
    
    elif args.dataset ==  "ca_housing":
        import pandas as pd
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        house_dataset = fetch_california_housing()

        df = pd.DataFrame(
            house_dataset.data,
            columns=house_dataset.feature_names
        )
        df.loc[:,"Price"] = house_dataset.target

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df.loc[:,:] = scaler.fit_transform(df)
        
        col_feature = house_dataset.feature_names
        col_target = ["Price"]

        y = df[col_target]
        x = df[col_feature]
        x = x.fillna(0)
        target_num = 1
        args.feature_dim = 8
        feature_train, feature_test, train_target, test_target = train_test_split(x, y, test_size=0.2)
        
    elif args.dataset == 'criteo':
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler
        args.task = "classification"
        args.feature_dim = 39
        class_num = 2
        col_target = ["label"]
        col_dense = [f"I{i}" for i in range(1,14)]
        col_sparse = [f"C{i}" for i in range(1,27)]
        
        #df = pd.read_csv("criteo_small.csv")
        df = pd.read_csv("criteo_medium.csv")
        
        df[col_sparse] = df[col_sparse].fillna('-1', )
        df[col_dense] = df[col_dense].fillna(0,)

        for feat in col_sparse:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
            
        scaler = StandardScaler()
        df[col_dense] = scaler.fit_transform(df[col_dense])
        
        y = df[col_target]
        x = df[col_dense + col_sparse]
        #x = df[col_dense]
        #args.feature_dim = 13
        
        feature_train, feature_test, train_target, test_target = train_test_split(x, y, test_size=0.2)
        
        
    elif args.dataset == 'kdd99':
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import fetch_kddcup99
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        class_num = 23
        args.feature_dim = 41
        dataset = fetch_kddcup99()

        col_sparse = ['protocol_type','service','flag']
        col_dense = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',  'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
        label = ['label']     
        df = pd.DataFrame(
            dataset.data,
            columns=dataset.feature_names
        )
        df.loc[:,label] = dataset.target  
    
          
        df[col_sparse] = df[col_sparse].fillna('-1', )
        df[col_dense] = df[col_dense].fillna(0,)
        
        for feat in col_sparse:
            lbe = LabelEncoder()
            df.loc[:,feat] = lbe.fit_transform(df[feat])
            #print(df[feat].value_counts())
        lbe = LabelEncoder()
        df.loc[:,label] = lbe.fit_transform(df[label])
        
        scaler = StandardScaler()
        df[col_dense] = scaler.fit_transform(df[col_dense])
        
        y = df[label]
        #x = df[col_dense + col_sparse].astype('float64')
        x = df[col_dense[:]].astype('float64')
        args.feature_dim = x.shape[1]
        
        feature_train, feature_test, train_target, test_target = train_test_split(x, y, test_size=0.2)
        
    elif args.dataset == 'ailerons':
        from mit_d3m import load_dataset
        import pandas as pd
        from sklearn.model_selection import train_test_split
        args.task = "regression"
        args.feature_dim = 40
        target_num = 1
        dataset = load_dataset('LL0_296_ailerons')

        col_feature = dataset.X.columns[1:]
        #col_target= dataset.y.columns[:]

        y = dataset.y.to_frame()
        x = dataset.X[col_feature]
        x = x.fillna(0)
        feature_train, feature_test, train_target, test_target = train_test_split(x, y, test_size=0.2)
        
    elif args.dataset == 'paint':
        import pandas as pd
        from sklearn.model_selection import train_test_split
        args.task = "regression"
        args.feature_dim = 125
        target_num = -1
        df = pd.read_csv('./2022-train-v2.csv')

        c = 0
        col_target = df.columns[c:c+1]
        col_feature1 = df.columns[6:33].to_list() # 27 cols
        col_feature2 = df.columns[33:43].to_list() # 10 cols
        col_feature3 = df.columns[43:103].to_list() # 60 cols
        col_feature4 = df.columns[103:].to_list() # 28 cols
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df.loc[:,:] = scaler.fit_transform(df)
        
        y = df[col_target]
        x = df[col_feature1 + col_feature2 + col_feature3 + col_feature4]
        x = x.fillna(0)
        feature_train, feature_test, train_target, test_target = train_test_split(x, y, test_size=0.2)

    else:
        from datasets import load_dataset
        train_data = load_dataset(args.dataset, split='train')
        test_data = load_dataset(args.dataset, split='test')

        if args.dataset == 'dbpedia_14':
            tf = 'content'
            class_num = 14
        elif args.dataset == 'ag_news':
            tf = 'text'
            class_num = 4
        elif args.dataset == 'banking77':
            tf = 'text'
            class_num = 77
        elif args.dataset == 'emotion':
            tf = 'text'
            class_num = 6
        elif args.dataset == 'rotten_tomatoes':
            tf = 'text'
            class_num = 2
        elif args.dataset == 'yelp_review_full':
            tf = 'text'
            class_num = 5
        elif args.dataset == 'sst2':
            tf = 'sentence'
            class_num = 2
            test_data = load_dataset(args.dataset, split='validation')

        train_text = [b[tf] for b in train_data]
        test_text = [b[tf] for b in test_data]
        train_label = [b['label'] for b in train_data]
        test_label = [b['label'] for b in test_data]
        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]

        vocab = create_vocab(clean_train)

    if args.task == "text":
        trainset = Textset(clean_train, train_label, vocab, args.max_len)
        testset = Textset(clean_test, test_label, vocab, args.max_len)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = trainset.collate, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = testset.collate)

        return train_loader, test_loader, class_num, vocab
    
    elif args.task == "regression" :
        trainset = StructDataset(feature_train, train_target)
        testset = StructDataset(feature_test, test_target)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = trainset.collate, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = testset.collate)
        
        return train_loader, test_loader, target_num
    
    elif args.task == "classification":
        trainset = StructDataset(feature_train, train_target)
        testset = StructDataset(feature_test, test_target)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = trainset.collate, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = testset.collate)
        
        return train_loader, test_loader, class_num

def train(model, data_loader, epoch, task="text"):
    if task == "text" or task == "classification":
        model.train()
        cor, num, tot_loss = 0, 0, []
        data_loader = tqdm(data_loader)
        for step, (x, y) in enumerate(data_loader):
            
            x, y = x.cuda(), y.cuda()
            losses = model(x, y)
            tot_loss.append(losses)
                
            pred = model.inference(x)
            #print(pred ,y)
            #print((pred.argmax(-1) == y).sum())
            #print(x.size(0))
            cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
            num += x.size(0)
            
            data_loader.set_description(f'Train {epoch} | Acc {cor/num} ({cor}/{num})')
        
        train_acc = cor/num
        train_loss = numpy.sum(tot_loss, axis=0)
        model.history["train_acc"].append(train_acc)
        model.history["train_loss"].append(train_loss)
        print(train_loss)
        print(f'Train Epoch{epoch} Acc {train_acc} ({cor}/{num})')
        
    elif task == "regression":
        model.train()
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
        
        print(f'Train Epoch{epoch} out_loss {train_out}, R2 {train_r2}')

def test(model, data_loader, shortcut=None, task="text"):
    if task == "text" or task == "classification":
        model.eval()
        cor, num = 0, 0
        #data_loader = tqdm(data_loader)
        for x, y in data_loader:
            x, y = x.cuda(), y.cuda()
            pred = model.inference(x, shortcut)
            #cor += (pred.argmax(-1) == y).sum().item()
            cor += (pred.argmax(-1).view(-1) == y.view(-1)).sum().item()
            num += x.size(0)

        return cor/num
    
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
            
        valid_out = mse_loss(y_out, y_tar).item()
        valid_r2 = r2_score(y_out, y_tar).item()
        #return torch.sqrt(out_loss/num).item()
        return valid_out, valid_r2

def predicting_for_sst(args, model, vocab):

    test_data = load_dataset('sst2', split='test')
    test_text = [b['sentence'] for b in test_data]
    test_label = [b['label'] for b in test_data]
    clean_test = [data_preprocessing(t, True) for t in test_text]
    
    testset = Textset(clean_test, test_label, vocab, args.max_len)
    test_loader = DataLoader(testset, batch_size=1, collate_fn = testset.collate)

    all_pred = []
    all_idx = []
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda()
        pred = model.inference(x).argmax(1).squeeze(0)
        all_pred.append(pred.item())
        all_idx.append(i)
    
    pred_file = {'index':all_idx, 'prediction':all_pred}
    output = pd.DataFrame(pred_file)
    output.to_csv('SST-2.tsv', sep='\t', index=False)

def main():

    
    args = get_args()
    path_name = f"{args.dataset}_{args.model}_l{str(args.num_layer)}"
    
    if args.task == "text":
        train_loader, test_loader, class_num, vocab = get_data(args)
        word_vec = get_word_vector(vocab, args.word_vec)
        
        if args.model == 'lstmal':
            model = LSTMModelML(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
        elif args.model == 'linearal':
            model = LinearModelML(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
        elif args.model == 'transformeral':
            model = TransformerModelML(vocab_size=len(vocab), num_layer=args.num_layer, emb_dim=args.emb_dim, l1_dim=args.l1_dim, lab_dim=args.label_emb, class_num=class_num, word_vec=word_vec, lr=args.lr)
    
    elif args.task == "classification":
        train_loader, test_loader, class_num  = get_data(args)
        if args.model == 'linearal':
            model = LinearALCLS(num_layer=args.num_layer, feature_dim=args.feature_dim, class_num=class_num, l1_dim=args.l1_dim, lab_dim=args.label_emb, lr=args.lr)
        elif args.model == 'linearalside' :
            model = LinearALsideCLS(num_layer=args.num_layer, side_dim=[5,5,5,5,5,5,5,3], class_num=class_num, l1_dim=args.l1_dim, lab_dim=args.label_emb, lr=args.lr)
    elif args.task == "regression":
        train_loader, test_loader, target_num  = get_data(args)
        if args.model == 'linearal':
            model = LinearALRegress(num_layer=args.num_layer, feature_dim=args.feature_dim, target_dim=1, l1_dim=args.l1_dim, lab_dim=args.label_emb, lr=args.lr)
        
    model = model.cuda()

    print('Start Training')
    
    if args.task == "text" or args.task == "classification":
        best_acc = 0
        for epoch in range(args.epoch):
            train(model, train_loader, epoch, task=args.task)
            valid_acc = []
            for layer in range(model.num_layer):
                result = test(model, test_loader, shortcut=layer+1, task=args.task)
                valid_acc.append(result)
                print(f'Test Epoch{epoch} layer{layer} Acc {result}')
                if result >= best_acc:
                    best_acc = result
                    torch.save(model.state_dict(), args.save_dir+f'/{path_name}.pt')
            model.history["valid_acc"].append(valid_acc)

        print('Best acc', best_acc)
        print('train_loss', numpy.array(model.history["train_loss"]).T.shape)
        print('valid_acc', numpy.array(model.history["valid_acc"]).T.shape)
        print('train_acc', numpy.array(model.history["train_acc"]).shape)
        
    elif args.task == "regression":
        best_r2, best_layer = 0,-1
        for epoch in range(args.epoch):
            train(model, train_loader, epoch, task="regression")
            valid_out, valid_r2 = [],[]
            
            for layer in range(model.num_layer):
                loss, r2 = test(model, test_loader, shortcut=layer+1, task="regression")
                valid_out.append(loss)
                valid_r2.append(r2)
                print(f'Test Epoch{epoch} layer{layer} out_loss {loss}, R2 {r2}')
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_layer = layer
                    torch.save(model.state_dict(), args.save_dir+f'/{path_name}.pt')
                    
            model.history["valid_out"].append(valid_out)
            model.history["valid_r2"].append(valid_r2)
            
        print(f'Best r2 {best_r2} at L{best_layer}' )
            
    plotResult(model,'result/'+ path_name, args.task)
    
    if args.dataset == 'sst2':
        model.load_state_dict(torch.load(args.save_dir+f'/{path_name}.pt'))
        predicting_for_sst(args, model, vocab)

    print('Start Testing')
    if args.task == "text" or args.task == "classification":
        model.load_state_dict(torch.load(args.save_dir+f'/{path_name}.pt'))
        model.eval()
        for layer in range(model.num_layer):
            y_out, y_tar = torch.Tensor([]),torch.Tensor([])
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                pred = model.inference(x, layer+1)
                y_out = torch.cat((y_out, pred.argmax(-1).view(-1).cpu()), 0)
                y_tar = torch.cat((y_tar, y.view(-1).cpu()), 0)
                
            plotConfusionMatrix(y_out, y_tar, [str(i) for i in range(23)], 'result/'+ f"{path_name}_test_l{layer}")
        
        
    
main()

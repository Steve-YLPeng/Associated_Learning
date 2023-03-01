from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import ConstantPad1d
import re
from nltk.corpus import stopwords
import string
import itertools
from collections import Counter
from itertools import count
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import confusion_matrix    
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, fetch_kddcup99
from sklearn.preprocessing import LabelEncoder, StandardScaler

stop_words = set(stopwords.words('english'))

def get_data(args):

    if args.dataset == 'imdb':
                
        class_num = 2
        df = pd.read_csv('./IMDB_Dataset.csv')
        df['cleaned_reviews'] = df['review'].apply(data_preprocessing, True)
        #df["seq_lenth"] = df["cleaned_reviews"].apply(lambda x:len(x.split()))
        #df = df[df["seq_lenth"]>100]
        corpus = [word for text in df['cleaned_reviews'] for word in text.split()]
        text = [t for t in df['cleaned_reviews']]
        label = []
        for t in df['sentiment']:
            if t == 'negative':
                label.append(1)
            else:
                label.append(0)
        vocab = create_vocab(corpus)
        
        clean_train, clean_test, train_label, test_label = train_test_split(text, label, test_size=0.2, random_state=35)
        clean_valid, clean_test, valid_label, test_label = train_test_split(clean_test, test_label, test_size=0.5, random_state=35)
    
    elif args.dataset ==  "ca_housing":

        house_dataset = fetch_california_housing()

        df = pd.DataFrame(
            house_dataset.data,
            columns=house_dataset.feature_names
        )
        df.loc[:,"Price"] = house_dataset.target

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
        
        # label encoded by value_count
        label_used = df['label'].value_counts(sort=True).index
        label_map = {v:i for i,v in enumerate(label_used)}
        df.loc[:,label] = df['label'].map(label_map)
        
        col_sparse_hot = []
        df[col_sparse]
        for col in col_sparse:
            onehot = pd.get_dummies(df[col], prefix=col)
            col_sparse_hot += onehot.columns.to_list()
            df = pd.concat((df,onehot), axis=1)
        df = df.drop(col_sparse,axis=1)
        
        scaler = StandardScaler()
        df[col_dense] = scaler.fit_transform(df[col_dense])
        
        # only keep top11 target label
        df = df[df['label'] < 11]
        class_num = 11
        
        data_train, data_test = pd.DataFrame(), pd.DataFrame()
        for _,group in df.groupby(label):
            train, test = train_test_split(group, test_size=0.2, random_state=3535)
            data_train = pd.concat((data_train,train))
            data_test = pd.concat((data_test,test))
        feature_train = data_train[col_dense[:]].astype('float64')
        feature_test = data_test[col_dense[:]].astype('float64')
        #feature_train = data_train[col_dense+col_sparse_hot].astype('float64')
        #feature_test = data_test[col_dense+col_sparse_hot].astype('float64')   
        train_target = data_train[label]
        test_target = data_test[label]
        args.feature_dim = feature_train.shape[1]
        del df
        
        
        
    elif args.dataset == 'ailerons':
        
        from mit_d3m import load_dataset
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
        
        scaler = StandardScaler()
        df.loc[:,:] = scaler.fit_transform(df)
        
        y = df[col_target]
        x = df[col_feature1 + col_feature2 + col_feature3 + col_feature4]
        x = x.fillna(0)
        feature_train, feature_test, train_target, test_target = train_test_split(x, y, test_size=0.2)

    else:
        from datasets import load_dataset
        train_data = load_dataset(args.dataset, split='train')
        #valid_data = load_dataset(args.dataset, split='test[:50%]')
        #test_data = load_dataset(args.dataset, split='test[50%:]')
        test_data = load_dataset(args.dataset, split='test').shuffle(seed=35)

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
            test_data = load_dataset(args.dataset, split='validation').shuffle(seed=35)
            
        # shuffle and split testset to 50/50
        split_size = int(test_data.num_rows*0.5)
        valid_data = test_data[:split_size]
        test_data = test_data[split_size:]
        
        train_text = [b[tf] for b in train_data]
        #test_text = [b[tf] for b in test_data]
        #valid_text = [b[tf] for b in valid_data]
        test_text = test_data[tf]
        valid_text = valid_data[tf]
        
        train_label = [b['label'] for b in train_data]
        #test_label = [b['label'] for b in test_data]
        #valid_label = [b['label'] for b in valid_data]
        test_label = test_data['label']
        valid_label = valid_data['label']
        
        clean_train = [data_preprocessing(t, True) for t in train_text]
        clean_test = [data_preprocessing(t, True) for t in test_text]
        clean_valid = [data_preprocessing(t, True) for t in valid_text]

        vocab = create_vocab(clean_train)

    if args.task == "text":
        trainset = Textset(clean_train, train_label, vocab, args.max_len)
        testset = Textset(clean_test, test_label, vocab, args.max_len)
        validset = Textset(clean_valid, valid_label, vocab, args.max_len)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = trainset.collate, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = testset.collate)
        valid_loader = DataLoader(validset, batch_size=args.batch_size, collate_fn = validset.collate)

        print(f"train size {len(trainset)}, valid size {len(validset)}, test size {len(testset)}")
        return train_loader, valid_loader, test_loader, class_num, vocab
    
    elif args.task == "regression" :
        trainset = StructDataset(feature_train, train_target)
        testset = StructDataset(feature_test, test_target)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = trainset.collate, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = testset.collate)
        
        return train_loader, valid_loader, test_loader, target_num
    
    elif args.task == "classification":
        trainset = StructDataset(feature_train, train_target)
        testset = StructDataset(feature_test, test_target)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = trainset.collate, shuffle=True)
        test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = testset.collate)
        
        return train_loader, valid_loader, test_loader, class_num


def plotConfusionMatrix(y_pred, y_true, label_name, save_filename=''):
    

    labels = label_name
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    df_cm = pd.DataFrame(cm, range(len(cm)), range(len(cm)))
    # plt.figure(figsize=(10,7))
    #sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size":6})
    #sn.heatmap(df_cm, annot=True,vmin=0, vmax=500, fmt='d', annot_kws={"size":6})
    
    #sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size":6})
    plt.savefig(save_filename+"_cm.png")
    plt.show()
    return
    
    
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
            #plt.plot(acc, label='valid acc L'+str(idx+1))
            plt.plot(acc, label='valid acc t%.1f'%(0.1*(idx+1)))
        plt.plot(history["train_acc"], "k", label='train_acc' )
        #plt.ylim(0.9, 1.0)
        plt.legend()
        plt.savefig(save_filename+"_acc.png")
        plt.show()
        
        # plot epoch AUC
        epoch_valid_AUC = numpy.array(history["valid_AUC"]).T
        for idx,AUC in enumerate(epoch_valid_AUC):
            #plt.plot(AUC, label='valid AUC L'+str(idx+1))
            plt.plot(AUC, label='valid AUC t%.1f'%(0.1*(idx+1)))
        plt.plot(history["train_AUC"], "k", label='train_AUC' )
        #plt.ylim(0.9, 1.0)
        plt.legend()
        plt.savefig(save_filename+"_AUC.png")
        plt.show()
        
        # plot epoch entropy
        epoch_valid_entr = numpy.array(history["valid_entr"]).T
        for idx,entr in enumerate(epoch_valid_entr):
            #plt.plot(entr, label='valid entr L'+str(idx+1))
            plt.plot(entr, label='valid entr t%.1f'%(0.1*(idx+1)))
        #plt.plot(history["train_entr"], "k", label='train_entr' )
        #plt.ylim(0.9, 1.0)
        plt.legend()
        plt.savefig(save_filename+"_entr.png")
        plt.show()
        
    if task == "regression":
        epoch_valid_out = numpy.array(history["valid_out"]).T
        for idx,out_loss in enumerate(epoch_valid_out):
            plt.plot(out_loss, label='valid out_loss L'+str(idx+1))
        plt.plot(history["train_out"], "k", label='train_out' )
        plt.legend()
        plt.savefig(save_filename+"_out.png")
        plt.show()
        
        epoch_valid_r2 = numpy.array(history["valid_r2"]).T
        for idx,r2 in enumerate(epoch_valid_r2):
            plt.plot(r2, label='valid_r2 L'+str(idx+1))
        
        plt.plot(history["train_r2"], "k", label='train_r2' )
        plt.legend()
        plt.savefig(save_filename+"_r2.png")
        plt.show()
        
def get_word_vector(vocab, emb='glove'):

    if emb == 'glove':
        fname = 'glove.6B.300d.txt'
        
        with open(fname,'rt') as fi:
            full_content = fi.read().strip().split('\n')

        data = {}
        for i in tqdm(range(len(full_content)), total=len(full_content), desc = 'loading glove vocabs...'):
            i_word = full_content[i].split(' ')[0]
            if i_word not in vocab.keys():
                continue
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            data[i_word] = i_embeddings

    elif emb == 'fasttext':
        fname = 'wiki-news-300d-1M.vec'

        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}

        for line in tqdm(fin, total=1000000, desc='loading fasttext vocabs...'):
            tokens = line.rstrip().split(' ')
            if tokens[0] not in vocab.keys():
                continue
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
    
    else:
        raise Exception('emb not implemented')

    w = []
    find = 0
    for word in vocab.keys():
        try:
            w.append(torch.tensor(data[word]))
            find += 1
        except:
            w.append(torch.rand(300))

    print('found', find, 'words in', emb)
    return torch.stack(w, dim=0)

def data_preprocessing(text, remove_stopword=False):

    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    if remove_stopword:
        text = [word for word in text.split() if word not in stop_words]
    else:
        text = [word for word in text.split()]
    text = ' '.join(text)
    if len(text) == 0:
        text = "<unk>"
    return text

def create_vocab(corpus, vocab_size=30000):

    corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()

    if vocab_size > len(sorted_words):
        v = len(sorted_words)
    else:
        v = vocab_size - 2

    vocab_to_int = {w: i + 2 for i, (w, c) in enumerate(sorted_words[:v])}

    vocab_to_int['<pad>'] = 0
    vocab_to_int['<unk>'] = 1
    print('vocab size', len(vocab_to_int))

    return vocab_to_int

class StructDataset(Dataset):
    def __init__(self, feature, target):
        super().__init__()
        self.x = feature.to_numpy()
        self.y = target.to_numpy()
        
    def collate(self, batch):
        x = torch.tensor([x for x,y in batch], dtype=torch.float32)
        y = [y for x,y in batch]
        y = torch.tensor(y, dtype=torch.float32)
        return x, y    
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

class Textset(Dataset):
    def __init__(self, text, label, vocab, max_len):
        super().__init__()

        new_text = []
        for t in text:
            words = t.split()
            if len(words) > max_len:
                text = " ".join(words[:max_len])
                new_text.append(text)
            else:
                text = " ".join(words[:]+["<pad>"]*(max_len-len(words)))
                new_text.append(text)
            """
            if len(t) > max_len:
                    t = t[:max_len]
                    new_text.append(t)
                else:
                    new_text.append(t)
            """
            
        self.x = new_text
        self.y = label
        self.vocab = vocab
        self.max_len = max_len
    def collate(self, batch):
        
        x = [torch.tensor(x) for x,y in batch]
        y = [y for x,y in batch]
        #x[0] = ConstantPad1d((0, self.max_len - x[0].shape[0]), 0)(x[0])

        x_tensor = pad_sequence(x, batch_first=True)
        #print(x_tensor)
        y = torch.tensor(y)
        return x_tensor, y

    def convert2id(self, text):
        r = []
        for word in text.split():
            if word in self.vocab.keys():
                r.append(self.vocab[word])
            else:
                r.append(self.vocab['<unk>'])
        return r
    
    def __getitem__(self, idx):
        text = self.x[idx]
        word_id = self.convert2id(text)
        return word_id, self.y[idx]

    def __len__(self):
        return len(self.x)

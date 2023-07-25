# Associated Learning

## 1.  Datasets
- For AGNews and DBpedia, dataset will be automatically downloaded during the training.
- For IMDB, please download the dataset from [Here](https://drive.google.com/file/d/1GRyOQs6TT0IXKDyha6zNjinmvREKyeuV/view)
- For cifar10, cifar100, and tinyImageNet, dataset will be automatically downloaded during the training
## 2. Word Embeddings
- We uses pretrained embeddings in our experiments, please download GloVe, Fasttext with the following commands
```bash=
wget https://nlp.stanford.edu/data/glove.6B.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip glove.6B.zip
unzip wiki-news-300d-1M.vec.zip
```
## 3. Requirements Setup
We developed the codes under Python 3.9.12 in Ubuntu. Install the required packages by
```bash=
mkdir ckpt
pip install -r requirements.txt
```

## 4. Execution
For RNN AL models training, please run
```bash=
python3 train_rnn.py \
--dataset <DATASET> --max-len <MAX_LEN> \
--model <MODEL> \
--epoch 20 \
--num-layer 4 \
--batch-train 256 --batch-test 512 \
--lr 0.0001 \
--l1-dim 300 --label-emb 128 \
--save-dir ./ckpt/  
```
The parameter settings are as follows
- dataset: can be one of ```ag_news```, ```imdb```, ```dbpedia_14```
- max-len: max lenth of input sequence for each dataset is ```177```, ```500```,and ```80``` respectively
- model: can be one of ```lstmal```, ```linearal```, ```transformeral```

For CNN AL models training, please run
```bash=
python3 train_cnn.py \
--dataset <DATASET> \
--model <MODEL> \
--epoch 100 \
--batch-train 128 --batch-test 1024 \
--lr 0.0001 \
--label-emb 500 \
--save-dir ./ckpt/ 
```
The parameter settings are as follows
- dataset: can be one of ```cifar10```, ```cifar100```, ```tinyImageNet```
- model: can be one of ```CNN_AL```, ```VGG_AL```, ```resnet_AL```
- For cnn AL models, num-layer is fixed to ```4```

The programs will train models based on the given parameters and will report the test accuracy for Shortcut/Adaptive inference each epoch.


For Dynamic Layer Accumulation experiment, please run
```bash=
python3 train_rnn_lbl.py \
--dataset <DATASET> --max-len <MAX_LEN> \
--model <MODEL> \
--epoch 80 \
--num-layer 4 \
--batch-train 256 --batch-test 512 \
--lr 0.0001 \
--l1-dim 300 --label-emb 128 \
--save-dir ./ckpt/  
```
```bash=
python3 train_cnn_lbl.py \
--dataset <DATASET> \
--model <MODEL> \
--epoch 400 \
--batch-train 128 --batch-test 1024 \
--lr 0.0001 \
--label-emb 500 \
--save-dir ./ckpt/ 
```
The programs will train models layer-by-layer and will report the test accuracy for Shortcut/Adaptive inference each epoch. Each layer of the models is trained for ```epoch/num-layer``` epochs. 
The parameter settings are the same as in the previous. 


For SideInput experiment (IMDB as example), please run
```bash=
python3 train_rnn.py \
--dataset imdb \
--model transformeralside \
--epoch 20 \
--max-len 400 \
--num-layer 4 \
--batch-train 256 --batch-test 512 \
--lr 0.0001 \
--l1-dim 300 --label-emb 128 \
--save-dir ./ckpt/ \
--side-dim 100-100-100-100
```
The parameter settings are as follows
- dataset: can be one of ```ag_news```, ```imdb```, ```dbpedia_14```
- model: the available options are ```lstmalside```, ```linearalside```, ```transformeralside```
- side-dim: specify the sequence length of each layer's input, separated by ```-``` , and their sum must be equal to max-len

The programs will train models with the sideinput setting and will report the test accuracy for Shortcut/Adaptive inference each epoch.

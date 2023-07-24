
# Associated Learning

## 1.  Datasets
- For AGNews and DBpedia, dataset will be automatically downloaded during the training.
- For IMDB, please download the dataset from [Here](https://drive.google.com/file/d/1GRyOQs6TT0IXKDyha6zNjinmvREKyeuV/view)
- For cifar10, cifar100, and tinyImageNet, dataset will be automatically downloaded during the training
## 2. Word Embeddings
- We uses pretrained embeddings in our experiments, please download GloVe, Fasttext with the following commands
```linux=
wget https://nlp.stanford.edu/data/glove.6B.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip glove.6B.zip
unzip wiki-news-300d-1M.vec.zip
```
## 3. Requirements Setup
We developed the codes under Python 3.9.12 in Ubuntu. Install the required packages by
```linux=
mkdir ckpt
pip install -r requirements.txt
```

## 4. Execution
### Training RNN models
```linux=
python3 train_rnn.py \
--dataset ag_news --max-len 177 \
--model transformeral \
--epoch 20 \
--num-layer 4 \
--batch-train 256 --batch-test 512 \
--lr 0.0001 \
--l1-dim 300 --label-emb 128 \
--save-dir ./ckpt/  
```
Config Settings
- dataset = {ag_news, imdb, dbpedia_14}
- max-len = {177, 500, 80}
- model = {lstmal, linearal, transformeral}

### Training CNN models

```linux=
python3 train_cnn.py \
--dataset cifar10 \
--model VGG_AL \
--epoch 100 \
--batch-train 128 --batch-test 1024 \
--lr 0.0001 \
--label-emb 500 \
--save-dir ./ckpt/ 
```
Config Settings
- dataset = {cifar10, cifar100, tinyImageNet}
- model = {CNN_AL, VGG_AL, resnet_AL}
- For cnn AL models, num-layer is fixed to 4

### Training RNN layer-by-layer 
```linux=
python3 train_rnn_lbl.py \
--dataset ag_news --max-len 177 \
--model transformeral \
--epoch 80 \
--num-layer 4 \
--batch-train 256 --batch-test 512 \
--lr 0.0001 \
--l1-dim 300 --label-emb 128 \
--save-dir ./ckpt/  
```
Config Settings
- dataset = {ag_news, imdb, dbpedia_14}
- max-len = {177, 500, 80}
- model = {lstmal, linearal, transformeral}

### Training CNN layer-by-layer 

```linux=
python3 train_cnn_lbl.py \
--dataset cifar10 \
--model VGG_AL \
--epoch 400 \
--batch-train 128 --batch-test 1024 \
--lr 0.0001 \
--label-emb 500 \
--save-dir ./ckpt/ 
```
Config Settings
- data = {cifar10, cifar100, tinyImageNet}
- model = {CNN_AL, VGG_AL, resnet_AL}

### Training RNN sideinput
```linux=
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
Config Settings
- dataset = {ag_news, imdb, dbpedia_14}
- model = {lstmalside, linearalside, transformeralside}
- side-dim: specify the sequence length of each layer's input, separated by ```-``` , and their sum must be equal to max-len 
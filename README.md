# Adaptive Inference and Dynamic Network Accumulation based on Associated Learning 

Associated learning (AL) is a new training alternative to end-to-end backpropagation.  This repo contains the codes to experiment with the new training and inference characteristics of AL -- Dynamic Layer Accumulation, Dynamic Feature Incorporation, and Adaptive Inference. 
  - Dynamic Layer Accumulation: adding new layers without disregarding any already-trained layers.
  - Dynamic Feature Incorporation: adding new features to the training data without retraining the entire network.
  - Adaptive Inference: providing multiple prediction paths during the inference stage, so adaptively deciding the inference path is possible.

We tested AL on the recurrent neural networks family (including LSTM and Transformer) and on the convolutional neural network family (including vanilla CNN, VGG, and ResNet) using AL instead of backpropagation.  

## 1.  Datasets
  - The AGNews, DBpedia, cifar10, cifar100, and tinyImageNet datasets will be automatically downloaded during the training if needed.
  - For IMDB, please download the dataset from [Here](https://drive.google.com/file/d/1GRyOQs6TT0IXKDyha6zNjinmvREKyeuV/view)

## 2. Word Embeddings
  - We use pretrained embeddings in our experiments. Please download GloVe and FastText with the following commands
  ```bash=
  wget https://nlp.stanford.edu/data/glove.6B.zip
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
  unzip glove.6B.zip
  unzip wiki-news-300d-1M.vec.zip
  ```

## 3. Requirements Setup
  - We develop the codes under Python 3.9.12 in Ubuntu. Install the required packages by
  ```bash=
  mkdir ckpt
  pip install -r requirements.txt
  ```

## 4. Execution

### Adaptive Inference

The programs will report the test accuracy for Shortcut/Adaptive inference for each epoch.

  - For RNN AL models training, please run
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

    - The parameter settings are as follows
      - dataset: can be one of ```ag_news```, ```imdb```, ```dbpedia_14```
      - max-len: max lenth of input sequence for each dataset is ```177```, ```500```,and ```80``` respectively
      - model: can be one of ```lstmal```, ```linearal```, ```transformeral```

  - For CNN AL models training, please run
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

    - The parameter settings are as follows
      - dataset: can be one of ```cifar10```, ```cifar100```, ```tinyImageNet```
      - model: can be one of ```CNN_AL```, ```VGG_AL```, ```resnet_AL```
      - For cnn AL models, num-layer is fixed to ```4```


### Dynamic Layer Accumulation

The programs will train models layer-by-layer and report the test accuracy for Shortcut/Adaptive inference at each epoch. Each layer of the models is trained for ```epoch/num-layer``` epochs. 
The parameter settings are the same as in the previous experiment. 

  - For the Dynamic Layer Accumulation experiment, please run

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

### Dynamic Feature Incorporation

The programs will train models with the side-input setting (the extra features are called the ``side-input'' here) and will report the test accuracy for Shortcut/Adaptive inference at each epoch.


  - For the side-input experiment (we use IMDB as an example), please run
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
  - The parameter settings are as follows
    - dataset: can be one of ```ag_news```, ```imdb```, ```dbpedia_14```
    - model: the available options are ```lstmalside```, ```linearalside```, ```transformeralside```
    - side-dim: specify the sequence length of each layer's input, separated by ```-``` , and their sum must be equal to max-len



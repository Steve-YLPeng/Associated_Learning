#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1115
# max len 
#   imdb/dbpidia 500
#   agnews 128
# emb dim 300
# L1 dim 300 
# batch 128

data = "dbpedia_14"
for model in ["linearal","lstmal","transformeral"]:
    layer = 5
    epoch = 10
    lr = 0.0001
    #for layer in range(1,11):
    for mask in range(1,layer+1):
        save = f"mask{mask}/"
        load = f"mask{mask-1}/"
        log = f"result/{data}/{data}_{model}_l{layer}_m{mask}_prefix.log"
        get_ipython().system('mkdir result/{data}/')
        get_ipython().system('mkdir ckpt/{save}/')
        get_ipython().system('mkdir ckpt/{save}/{data}/')
        if mask != 1:
            get_ipython().system('python3 dis_train_side.py --dataset {data} --model {model} --epoch {epoch}             --max-len 500             --num-layer {layer} --batch-size 128 --lr {lr} --l1-dim 300 --label-emb 128             --prefix-mask True             --lr-schedule plateau             --train-mask {mask}             --save-dir ./ckpt/{save}             --load-dir ./ckpt/{load}             --task text > {log}')
        else:
            get_ipython().system('python3 dis_train_side.py --dataset {data} --model {model} --epoch {epoch}             --max-len 500             --num-layer {layer} --batch-size 128 --lr {lr} --l1-dim 300 --label-emb 128             --prefix-mask True             --lr-schedule plateau             --train-mask {mask}             --save-dir ./ckpt/{save}             --task text > {log}')
        import gc
        gc.collect()


# In[ ]:





# In[1]:


# 1115
# max len 
#   imdb/dbpidia 500
#   agnews 128
# emb dim 300
# L1 dim 300 
# batch 128

data = "dbpedia_14"
for model in ["linearal","lstmal","transformeral"]:
    layer = 5
    epoch = 10
    lr = 0.0001
    #for layer in range(1,11):
    for mask in range(1,layer+1):
        save = f"mask{mask}/"
        load = f"mask{mask-1}/"
        log = f"result/{data}/{data}_{model}_l{layer}_m{mask}.log"
        get_ipython().system('mkdir result/{data}/')
        get_ipython().system('mkdir ckpt/{save}/')
        get_ipython().system('mkdir ckpt/{save}/{data}/')
        if mask != 1:
            get_ipython().system('python3 dis_train_side.py --dataset {data} --model {model} --epoch {epoch}             --max-len 500             --num-layer {layer} --batch-size 128 --lr {lr} --l1-dim 300 --label-emb 128             --lr-schedule plateau             --train-mask {mask}             --save-dir ./ckpt/{save}             --load-dir ./ckpt/{load}             --task text > {log}')
        else:
            get_ipython().system('python3 dis_train_side.py --dataset {data} --model {model} --epoch {epoch}             --max-len 500             --num-layer {layer} --batch-size 128 --lr {lr} --l1-dim 300 --label-emb 128             --lr-schedule plateau             --train-mask {mask}             --save-dir ./ckpt/{save}             --task text > {log}')
        import gc
        gc.collect()  


# In[2]:


# 1115
# max len 
#   imdb/dbpidia 500
#   agnews 128
# emb dim 300
# L1 dim 300 
# batch 128

data = "dbpedia_14"
for model in ["linearal","lstmal","transformeral"]:
    #for layer in range(1,11):
    for layer in [5]:
        log = f"result/{data}/{data}_{model}_l{layer}.log"
        get_ipython().system('mkdir result/{data}/')
        get_ipython().system('mkdir ckpt/{data}/')
        get_ipython().system('python3 dis_train_side.py --dataset {data} --model {model} --epoch 50         --max-len 500 --l1-dim 300 --label-emb 128         --num-layer {layer} --lr 0.0001 --task text > {log}')
    import gc
    gc.collect()  


# In[ ]:


# no shuffle


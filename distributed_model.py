import torch
import torch.nn as nn
from transformer.encoder import TransformerEncoder
from torch.nn import ModuleList
from typing import List
from torch.optim.lr_scheduler import ReduceLROnPlateau


def initialize_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
    elif isinstance(model, nn.LSTM):
        for name, param in model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    elif isinstance(model, TransformerEncoder):
        return
        for name, param in model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

### Layer component definition
class AE(nn.Module):
    
    ############################################
    #   g: Forward function for input Y 
    #       (Encoder of AutoEncoder)
    #   h: Decoder of AutoEncoder
    # cri: Loss function for AutoEncoder loss
    ############################################
    def __init__(self, inp_dim, out_dim, cri='ce', act=None):
        super().__init__()

        
        if act == None:
            self.g = nn.Sequential(
                    nn.Linear(inp_dim, out_dim),
                    nn.Tanh()            
                )
            if cri == 'ce':
                self.h = nn.Sequential(
                    nn.Linear(out_dim, inp_dim),
                    #nn.Tanh(),
                    nn.Sigmoid(),
                )
                self.cri = nn.CrossEntropyLoss()
                
            elif cri == 'mse':
                self.h = nn.Sequential(
                    nn.Linear(out_dim, inp_dim),
                )
                self.cri = nn.MSELoss()
            
        else:
            if act[0] != None:
                self.g = nn.Sequential(
                    nn.Linear(inp_dim, out_dim),       
                    act[0]    
                )
            else:
                self.g = nn.Sequential(
                    nn.Linear(inp_dim, out_dim),         
                )
            if act[1] != None:
                self.h = nn.Sequential(
                    nn.Linear(out_dim, inp_dim),
                    act[1] 
                )
            else:
                self.h = nn.Sequential(
                    nn.Linear(out_dim, inp_dim), 
                )
            if cri == 'mse' :
                self.cri = nn.MSELoss()
            else :
                self.cri = nn.CrossEntropyLoss()
            
        self.mode = cri
    
    def forward(self, x):
        enc_x = self.g(x)
        rec_x = self.h(enc_x)
        if self.mode == 'ce':
            loss = self.cri(rec_x, x.argmax(1))
        elif self.mode == 'mse':
            loss = self.cri(rec_x, x)
            
        return enc_x, loss
class ENC(nn.Module):
    
    ############################################
    #   f: Forward function for input X
    #   b: Bridge function
    # cri: Loss function for associated loss
    ############################################
    def __init__(self, inp_dim, out_dim, lab_dim=128, f='emb', n_heads=4, word_vec=None):
        super().__init__()
        
        self.b = nn.Sequential(
            nn.Linear(out_dim, lab_dim),
            nn.Tanh()
        )
        
        self.mode = f
        if f == 'emb':
            self.f = nn.Embedding(inp_dim, out_dim)
            if word_vec is not None:
                self.f = nn.Embedding.from_pretrained(word_vec, freeze=False)
        elif f == 'lstm':
            self.f = nn.LSTM(inp_dim, out_dim, bidirectional=True, batch_first=True)
        elif f == 'trans':
            self.f = TransformerEncoder(d_model=inp_dim, d_ff=out_dim, n_heads=n_heads)
            #self.f = nn.TransformerEncoderLayer(d_model=inp_dim, nhead=6)
            
            self.b = nn.Sequential(
                nn.Linear(inp_dim, lab_dim),
                nn.Tanh()
            )
        elif f == 'linear':
            self.f = nn.Sequential(
                nn.Linear(inp_dim, out_dim),
                #nn.BatchNorm1d(out_dim),
                nn.ELU(),
                #nn.Tanh(),
                #nn.Sigmoid(),
                
            )
            self.b = nn.Sequential(
                nn.Linear(out_dim, lab_dim),
                #nn.BatchNorm1d(out_dim),
                #nn.ELU(),
                nn.Tanh(),
            )

        self.cri = nn.MSELoss()
    
    def forward(self, x, tgt, mask=None, h=None):

        if self.mode == 'emb' :
            enc_x = self.f(x.long())
        elif self.mode == 'linear' :
            enc_x = self.f(x)
        elif self.mode == 'lstm':
            enc_x, (h, c) = self.f(x, h)
        elif self.mode == 'trans':
            enc_x = self.f(x, mask=mask)
            #print(mask)
            #enc_x = self.f(x, src_mask=mask)
            
            
        red_x = self.reduction(enc_x, mask, h)
        red_x = self.b(red_x)
        loss = self.cri(red_x, tgt)

        return enc_x, loss, h, mask

    def reduction(self, x, mask=None, h=None):

        # to match bridge function
        if self.mode == 'emb':
            return x.mean(1)

        elif self.mode == 'lstm':
            _h = h[0] + h[1]
            return _h

        elif self.mode == 'trans':

            denom = torch.sum(mask, -1, keepdim=True)
            feat = torch.sum(x * mask.unsqueeze(-1), dim=1) / denom
            return feat
        
        elif self.mode == 'linear':
            return x
        
        
### AL layers definition

class EMBLayer(nn.Module):
    def __init__(self, inp_dim, lab_dim, hid_dim, lr, class_num=None, word_vec=None, ae_cri='ce'):
        super().__init__()

        self.enc = ENC(inp_dim, hid_dim, lab_dim=lab_dim, f='emb', word_vec=word_vec)
        assert class_num is not None
        self.ae = AE(class_num, lab_dim, cri=ae_cri)
    
        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr)
        self.ae_sche = ReduceLROnPlateau(self.ae_opt, mode="max", factor=0.5, patience=3)
        self.enc_sche = ReduceLROnPlateau(self.enc_opt, mode="max", factor=0.5, patience=3)
        
    def forward(self, x, y, mask=None, h=None):

        self.ae_opt.zero_grad()
        enc_y , ae_loss = self.ae(y)
        if self.training:
            ae_loss.backward()
            #nn.utils.clip_grad_norm_(self.ae.parameters(), 5)
            self.ae_opt.step()
    
        self.enc_opt.zero_grad()
        tgt = enc_y.clone().detach()
        enc_x, enc_loss, hidden, mask = self.enc(x, tgt, mask, h)
        if self.training:
            enc_loss.backward()
            #nn.utils.clip_grad_norm_(self.enc.parameters(), 5)
            self.enc_opt.step()

        return enc_x.detach(), enc_y.detach(), ae_loss, enc_loss, [hidden, mask]
        return enc_x.clone().detach(), enc_y.clone().detach(), ae_loss, enc_loss, [hidden, mask]

class TransLayer(nn.Module):
    def __init__(self, inp_dim, lab_dim, hid_dim, lr, out_dim=None, ae_cri='mse', ae_act=None):
        super().__init__()

        self.enc = ENC(inp_dim, hid_dim, lab_dim=lab_dim, f='trans')
        if out_dim == None:
            self.ae = AE(lab_dim, lab_dim, cri=ae_cri, act=ae_act)
        else:
            self.ae = AE(out_dim, lab_dim, cri=ae_cri, act=ae_act)

        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr)
        self.ae_sche = ReduceLROnPlateau(self.ae_opt, mode="max", factor=0.5, patience=3)
        self.enc_sche = ReduceLROnPlateau(self.enc_opt, mode="max", factor=0.5, patience=3)
    
    def forward(self, x, y, mask):

        self.ae_opt.zero_grad()
        enc_y , ae_loss = self.ae(y)
        if self.training:
            ae_loss.backward()
            #nn.utils.clip_grad_norm_(self.ae.parameters(), 0.001)
            #nn.utils.clip_grad_value_(self.parameters(), 0.0001)
            self.ae_opt.step()
    
        self.enc_opt.zero_grad()
        tgt = enc_y.clone().detach()
        enc_x, enc_loss, h, mask = self.enc(x, tgt, mask=mask)
        if self.training:
            enc_loss.backward()
            #nn.utils.clip_grad_norm_(self.enc.parameters(), 0.001)
            #nn.utils.clip_grad_value_(self.parameters(), 0.0001)
            self.enc_opt.step()

        return enc_x.detach(), enc_y.detach(), ae_loss, enc_loss, mask        
        return enc_x.clone().detach(), enc_y.clone().detach(), ae_loss, enc_loss, mask        

class LSTMLayer(nn.Module):
    def __init__(self, inp_dim, lab_dim, hid_dim, lr, out_dim=None, ae_cri='mse', ae_act=None):
        super().__init__()

        self.enc = ENC(inp_dim, hid_dim, lab_dim=lab_dim, f='lstm')
        if out_dim == None:
            self.ae = AE(lab_dim, lab_dim, cri=ae_cri, act=ae_act)
        else:
            self.ae = AE(out_dim, lab_dim, cri=ae_cri, act=ae_act)
    
        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr)
        self.ae_sche = ReduceLROnPlateau(self.ae_opt, mode="max", factor=0.5, patience=3)
        self.enc_sche = ReduceLROnPlateau(self.enc_opt, mode="max", factor=0.5, patience=3)

    def forward(self, x, y, mask=None, h=None):

        self.ae_opt.zero_grad()
        enc_y , ae_loss = self.ae(y)
        if self.training:
            ae_loss.backward()
            #nn.utils.clip_grad_norm_(self.ae.parameters(), 5)
            self.ae_opt.step()
    
        self.enc_opt.zero_grad()
        tgt = enc_y.clone().detach()
        enc_x, enc_loss, hidden, _ = self.enc(x, tgt, mask, h)
        if self.training:
            enc_loss.backward()
            #nn.utils.clip_grad_norm_(self.enc.parameters(), 5)
            self.enc_opt.step()
        (h, c) = hidden
        h = h.reshape(2, x.size(0), -1)
        hidden = (h.clone().detach(), c.clone().detach())

        return enc_x.detach(), enc_y.detach(), ae_loss, enc_loss, [hidden, mask]
        return enc_x.clone().detach(), enc_y.clone().detach(), ae_loss, enc_loss, [hidden, mask]

class LinearLayer(nn.Module):
    def __init__(self, inp_dim, lab_dim, hid_dim, lr, out_dim=None, ae_cri='mse', ae_act=None):
        super().__init__()

        self.enc = ENC(inp_dim, hid_dim, lab_dim=lab_dim, f='linear')
        if out_dim == None:
            self.ae = AE(lab_dim, lab_dim, cri=ae_cri, act=ae_act)
        else:
            self.ae = AE(out_dim, lab_dim, cri=ae_cri, act=ae_act)
    
        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr)
        self.ae_sche = ReduceLROnPlateau(self.ae_opt, mode="max", factor=0.5, patience=3)
        self.enc_sche = ReduceLROnPlateau(self.enc_opt, mode="max", factor=0.5, patience=3)

    def forward(self, x, y):

        self.ae_opt.zero_grad()
        enc_y , ae_loss = self.ae(y)
        if self.training:
            ae_loss.backward()
            #nn.utils.clip_grad_norm_(self.ae.parameters(), 5)
            self.ae_opt.step()
    
        self.enc_opt.zero_grad()
        tgt = enc_y.clone().detach()
        enc_x, enc_loss, _, _ = self.enc(x, tgt)
        if self.training:
            enc_loss.backward()
            #nn.utils.clip_grad_norm_(self.enc.parameters(), 5)
            self.enc_opt.step()

        return enc_x.detach(), enc_y.detach(), ae_loss, enc_loss
        return enc_x.clone().detach(), enc_y.clone().detach(), ae_loss, enc_loss


###########################################################
# Definition of AL text classification models. 
# models:   
#   TransModel, LSTMModel, 
#   TransformerModelML, LSTMModelML, LinearModelMLdeta
###########################################################
class TransModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None):
        super().__init__()

        self.emb = EMBLayer(vocab_size, lab_dim, emb_dim, lr = 0.001, class_num=class_num, word_vec=word_vec)
        self.l1 = TransLayer(emb_dim, lab_dim, l1_dim, lr=lr)
        self.l1_dim = l1_dim
        self.l2 = TransLayer(emb_dim, lab_dim, l1_dim, lr=lr)
        self.losses = [0.0] * 6
        self.class_num = class_num
        
    def forward(self, x, y):

        mask = self.get_mask(x)
        y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        emb_x, emb_y, emb_ae, emb_as, _ = self.emb(x, y) # also updated

        l1_x, l1_y, l1_ae, l1_as, mask = self.l1(emb_x, emb_y, mask)
        l2_x, l2_y, l2_ae, l2_as, mask = self.l2(l1_x, l1_y, mask)

        return [emb_ae, emb_as, l1_ae, l1_as, l2_ae, l2_as]
    
    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()

    def inference(self, x):

        mask = self.get_mask(x)
        emb_x = self.emb.enc.f(x)
        l1_x = self.l1.enc.f(emb_x, mask)
        l2_x = self.l2.enc.f(l1_x, mask)

        denom = torch.sum(mask, -1, keepdim=True)
        feat = torch.sum(l2_x * mask.unsqueeze(-1), dim=1) / denom
        bridge = self.l2.enc.b(feat)

        _out = self.l2.ae.h(bridge)
        _out = self.l1.ae.h(_out)
        pred = self.emb.ae.h(_out)

        return pred

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None):
        super().__init__()

        self.emb = EMBLayer(vocab_size, lab_dim, emb_dim, lr = 0.001, class_num=class_num, word_vec=word_vec)
        self.l1 = LSTMLayer(emb_dim, lab_dim, l1_dim, lr=lr)
        self.l1_dim = l1_dim
        self.l2 = LSTMLayer(l1_dim*2, lab_dim, l1_dim, lr=lr)
        self.losses = [0.0] * 6
        self.class_num = class_num

    def forward(self, x, y):
        
        y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        emb_x, emb_y, emb_ae, emb_as, _ = self.emb(x, y) # also updated

        l1_x, l1_y, l1_ae, l1_as, [h, _] = self.l1(emb_x, emb_y)
        l1_x = torch.cat((l1_x[:, :, :self.l1_dim], l1_x[:, :, self.l1_dim:]), dim=-1)

        l2_x, l2_y, l2_ae, l2_as, [h, _] = self.l2(l1_x, l1_y, h)

        return [emb_ae, emb_as, l1_ae, l1_as, l2_ae, l2_as]
    
    def inference(self, x):
        emb_x = self.emb.enc.f(x)
        l1_x, (h, c) = self.l1.enc.f(emb_x)
        l1_x = torch.cat((l1_x[:, :, :self.l1_dim], l1_x[:, :, self.l1_dim:]), dim=-1)
        # print(l1_x.shape, )
        l2_x, (h, c) = self.l2.enc.f(l1_x, (h, c))
        h = h[0] + h[1]

        bridge = self.l2.enc.b(h)

        _out = self.l2.ae.h(bridge)
        _out = self.l1.ae.h(_out)
        pred = self.emb.ae.h(_out)

        return pred
       
### YLP: multi-layer version of al

class alModel(nn.Module):
    def __init__(self, num_layer, l1_dim, class_num, lab_dim, emb_dim=None):
        super().__init__()
        
        self.num_layer = num_layer
        self.history = {"train_acc":[],"valid_acc":[],"train_loss":[],
                        "train_AUC":[],"valid_AUC":[],
                        "train_r2":[],"valid_r2":[],
                        "train_out":[],"valid_out":[],
                        "train_entr":[],"valid_entr":[],} 
        self.emb_dim = emb_dim
        self.l1_dim = l1_dim
        self.lab_dim = lab_dim
        self.losses = [0.0] * (num_layer*2)
        self.class_num = class_num
        self.layers = ModuleList([])
        
    def schedulerStep(self, layer, score):
        #print("lr step")
        self.layers[layer].ae_sche.step(score)
        self.layers[layer].enc_sche.step(score)
        
    def train(self, is_train=True, layer_mask = None): 
        if layer_mask!=None:
            for layer in range(self.num_layer):
                if layer in layer_mask:
                    self.layers[layer].train()
                    for param in self.layers[layer].parameters():
                        param.requires_grad = True
                else:
                    self.layers[layer].eval()
                    for param in self.layers[layer].parameters():
                        param.requires_grad = False
        else:
            super().train(is_train)  

class TransformerModelML(alModel):    
    def __init__(self, vocab_size, num_layer, emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None):
        super().__init__(num_layer, l1_dim, class_num, lab_dim, emb_dim)
               
        layers = ModuleList([])
        for idx in range(self.num_layer):
            if idx == 0:
                layer = EMBLayer(vocab_size, lab_dim, emb_dim, lr = 0.001, class_num=class_num, word_vec=word_vec)
            #elif idx == 1:
            #    layer = TransLayer(emb_dim, lab_dim, l1_dim, lr=lr)
            else:
                layer = TransLayer(emb_dim, lab_dim, l1_dim, lr=lr)
            layers.append(layer)
        
        self.layers = layers     
        
    def forward(self, x, y):
        
        layer_loss = []
        mask = self.get_mask(x)
        y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        
        # forward function also update
        for idx,layer in enumerate(self.layers):
            if idx == 0:
                x_out, y_out, ae_out, as_out, _ = layer(x, y)
                layer_loss.append([ae_out.item(), as_out.item()])
            else:
                x_out, y_out, ae_out, as_out, mask = layer(x_out, y_out, mask)
                layer_loss.append([ae_out.item(), as_out.item()])    
        
        return layer_loss
    
    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        
        mask = self.get_mask(x)
        
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        for idx in range(len_path):
            if idx==0: # embed
                x_out = self.layers[idx].enc.f(x.long())          
            else:
                x_out = self.layers[idx].enc.f(x_out, mask)
                     
        # bridge        
        if len_path == 1:
            x_out = x_out.mean(1)  
        else:
            denom = torch.sum(mask, -1, keepdim=True)
            x_out = torch.sum(x_out * mask.unsqueeze(-1), dim=1) / denom
            
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out
    
    
class LSTMModelML(alModel):    
    def __init__(self, vocab_size, num_layer, emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None):
        super().__init__(num_layer, l1_dim, class_num, lab_dim, emb_dim)
             
        layers = ModuleList([])
        for idx in range(self.num_layer):
            if idx == 0:
                layer = EMBLayer(vocab_size, lab_dim, emb_dim, lr = 0.001, class_num=class_num, word_vec=word_vec)
            elif idx == 1:
                layer = LSTMLayer(emb_dim, lab_dim, l1_dim, lr=lr)
            else:
                layer = LSTMLayer(l1_dim*2, lab_dim, l1_dim, lr=lr)
            layers.append(layer)
        
        self.layers = layers     
        
    def forward(self, x, y):
        
        layer_loss = []
        y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        
        # forward function also update
        for idx,layer in enumerate(self.layers):
            if idx == 0:
                x_out, y_out, ae_out, as_out, _ = layer(x, y)
                layer_loss.append([ae_out.item(), as_out.item()])
            elif idx == 1:
                x_out, y_out, ae_out, as_out, [h, _] = layer(x_out, y_out)
                x_out = torch.cat((x_out[:, :, :self.l1_dim], x_out[:, :, self.l1_dim:]), dim=-1)
                layer_loss.append([ae_out.item(), as_out.item()])    
            
            else:
                x_out, y_out, ae_out, as_out, [h, _] = layer(x_out, y_out,h)
                x_out = torch.cat((x_out[:, :, :self.l1_dim], x_out[:, :, self.l1_dim:]), dim=-1)
                layer_loss.append([ae_out.item(), as_out.item()])
                
        return layer_loss
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        for idx in range(len_path):
            if idx==0:
                x_out = self.layers[idx].enc.f(x.long())
                  
                
            elif idx==1:
                x_out, (h, c) = self.layers[idx].enc.f(x_out)
                x_out = torch.cat((x_out[:, :, :self.l1_dim], x_out[:, :, self.l1_dim:]), dim=-1)
                
            else:
                x_out, (h, c) = self.layers[idx].enc.f(x_out, (h, c))
                x_out = torch.cat((x_out[:, :, :self.l1_dim], x_out[:, :, self.l1_dim:]), dim=-1)
                
                
        if len_path == 1:
            x_out = x_out.mean(1)  
        else:
            x_out = h[0] + h[1]
            
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out


class LinearModelML(alModel):    
    def __init__(self, vocab_size, num_layer, emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None):
        super().__init__(num_layer, l1_dim, class_num, lab_dim, emb_dim)
             
        layers = ModuleList([])
        for idx in range(self.num_layer):
            if idx == 0:
                layer = EMBLayer(vocab_size, lab_dim, emb_dim, lr = 0.001, class_num=class_num, word_vec=word_vec)
            elif idx == 1:
                layer = LinearLayer(emb_dim, lab_dim, l1_dim, lr=lr)
            else:
                layer = LinearLayer(l1_dim, lab_dim, l1_dim, lr=lr)
            layers.append(layer)
        
        self.layers = layers     
        
    def forward(self, x, y):
        
        layer_loss = []
        y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        
        # forward function also update
        for idx,layer in enumerate(self.layers):
            if idx == 0:
                x_out, y_out, ae_out, as_out, _ = layer(x, y)
                x_out = x_out.mean(1)
                layer_loss.append([ae_out.item(), as_out.item()])
                
            else:
                x_out, y_out, ae_out, as_out = layer(x_out, y_out)
                layer_loss.append([ae_out.item(), as_out.item()])
                
        return layer_loss
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        for idx in range(len_path):
            if idx==0:
                x_out = self.layers[idx].enc.f(x.long())
                #print("embed out", x_out.shape)
                x_out = x_out.mean(1)
                #print("embed out", _out.shape)
            else:
                x_out = self.layers[idx].enc.f(x_out)    
            
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out


###########################################################
# Definition of AL regression models. 
# models: 
#   LinearALReg, 
###########################################################

class LinearALRegress(alModel): 
    
    def __init__(self, num_layer, feature_dim, class_num, l1_dim, lr, lab_dim=128):
        super().__init__(num_layer, l1_dim, class_num, lab_dim)
              
        layers = ModuleList([])
        for idx in range(self.num_layer):
            if idx == 0:
                act = [nn.Tanh(),None]
                #act = [None,None]
                layer = LinearLayer(inp_dim=feature_dim, out_dim=class_num, 
                                    hid_dim=l1_dim, lab_dim=lab_dim, lr=lr, ae_act=act)
            else:
                layer = LinearLayer(l1_dim, lab_dim, l1_dim, lr=lr)
            layers.append(layer)
        
        self.layers = layers     


    def forward(self, x, y):
        
        layer_loss = []
        #y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        
        # forward function also update
        for idx,layer in enumerate(self.layers):
            if idx == 0:
                #print(x.shape)
                #print(y.shape)
                x_out, y_out, ae_out, as_out = layer(x, y)
                layer_loss.append([ae_out.item(), as_out.item()])
                
            else:
                x_out, y_out, ae_out, as_out = layer(x_out, y_out)
                layer_loss.append([ae_out.item(), as_out.item()])
                
        return layer_loss
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        for idx in range(len_path):
            if idx==0:
                x_out = self.layers[idx].enc.f(x)
            else:
                x_out = self.layers[idx].enc.f(x_out)    
            
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out
    
class LinearALCLS(alModel):    
    def __init__(self, num_layer, feature_dim, class_num, l1_dim, lr, lab_dim=128):
        super().__init__(num_layer, l1_dim, class_num, lab_dim)
              
        layers = ModuleList([])
        for idx in range(self.num_layer):
            if idx == 0:
                #act = [nn.Tanh(), nn.Tanh()]
                #act = [nn.ELU(), nn.ELU()]
                act = [nn.Tanh(),nn.Sigmoid()]
                layer = LinearLayer(inp_dim=feature_dim, out_dim=class_num, 
                                    hid_dim=l1_dim, lab_dim=lab_dim, lr=lr, ae_cri='ce', ae_act=act)
            else:
                layer = LinearLayer(l1_dim, lab_dim, l1_dim, lr=lr, ae_cri='mse')
            layers.append(layer)
        
        self.layers = layers     
        
    def forward(self, x, y):
        
        layer_loss = []
        y = torch.nn.functional.one_hot(y.view(-1).long(), self.class_num).float().to(y.device)

        # forward function also update
        for idx,layer in enumerate(self.layers):
            if idx == 0:
                x_out, y_out, ae_out, as_out = layer(x, y)
                layer_loss.append([ae_out.item(), as_out.item()])
            else:
                x_out, y_out, ae_out, as_out = layer(x_out, y_out)
                layer_loss.append([ae_out.item(), as_out.item()])
                
        return layer_loss
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        for idx in range(len_path):
            if idx==0:
                x_out = self.layers[idx].enc.f(x)
            else:
                x_out = self.layers[idx].enc.f(x_out)    
            
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out
    
class LinearALsideCLS(alModel):    
    def __init__(self, num_layer, side_dim:List[int], class_num, l1_dim, lr, lab_dim=128):
        super().__init__(num_layer, l1_dim, class_num, lab_dim)
        
        assert num_layer == len(side_dim)
        self.side_dim = side_dim
     
        layers = ModuleList([])
        for idx in range(self.num_layer):
            if idx == 0:
                #act = [nn.Tanh(), nn.Tanh()]
                #act = [nn.ELU(), nn.ELU()]
                act = [nn.Tanh(),nn.Sigmoid()]
                #act = None 
                layer = LinearLayer(inp_dim=side_dim[0], out_dim=class_num, 
                                    hid_dim=l1_dim, lab_dim=lab_dim, lr=lr, ae_cri='ce', ae_act=act)
            else:
                concat_dim = l1_dim + side_dim[idx]

                layer = LinearLayer(inp_dim=concat_dim, hid_dim=l1_dim, lab_dim=lab_dim, lr=lr, ae_cri='mse')
            layers.append(layer)
            
        self.layers = layers  
    
    def sidedata(self, x):
        return torch.split(x, self.side_dim, -1)
    
    def forward(self, x, y):
        
        layer_loss = []
        #print(y.shape)
        y = torch.nn.functional.one_hot(y.view(-1).long(), self.class_num).float().to(y.device)
        x_side = self.sidedata(x)
        
        # forward function also update
        for idx,layer in enumerate(self.layers):
            if idx == 0:
                x_out, y_out, ae_out, as_out = layer(x_side[idx], y)
                layer_loss.append([ae_out.item(), as_out.item()])
            else:
                x_cat = torch.cat((x_out, x_side[idx]), dim=-1)
                x_out, y_out, ae_out, as_out = layer(x_cat, y_out)
                layer_loss.append([ae_out.item(), as_out.item()])
                
        return layer_loss        
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        x_side = self.sidedata(x)
        for idx in range(len_path):
            if idx==0:
                x_out = self.layers[idx].enc.f(x_side[idx])
            else:
                x_cat = torch.cat((x_out, x_side[idx]), dim=-1)
                x_out = self.layers[idx].enc.f(x_cat)    
            
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out

class alSideModel(alModel): 
    def __init__(self, vocab_size, num_layer, side_dim:List[int], emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None, same_emb=False):
        super().__init__(num_layer, l1_dim, class_num, lab_dim, emb_dim)
        assert num_layer == len(side_dim)
        assert emb_dim == l1_dim
        self.side_dim = side_dim
        self.same_emb = same_emb
        
    def sidedata(self, x):
        return tuple(x for _ in range(self.num_layer))
        return torch.split(x, self.side_dim, -1)
    
class LinearALsideText(alSideModel): 
    def __init__(self, vocab_size, num_layer, side_dim:List[int], emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None, same_emb=False):
        super().__init__(vocab_size, num_layer, side_dim, emb_dim, l1_dim, lr, class_num, lab_dim, word_vec, same_emb)
        
        # emb
        if self.same_emb:
            if word_vec is not None:
                self.emb = nn.Embedding.from_pretrained(word_vec, freeze=False)
            else:
                self.emb = nn.Embedding(vocab_size, emb_dim) 
        else:
            emb_layers = ModuleList([])
            for idx in range(self.num_layer):
                if word_vec is not None:
                    emb = nn.Embedding.from_pretrained(word_vec, freeze=False)
                else:
                    emb = nn.Embedding(vocab_size, emb_dim) 
                emb_layers.append(emb)
            self.emb_layers = emb_layers
          
        # layers define  
        layers = ModuleList([])
        for idx in range(self.num_layer):
            if idx == 0:
                layer = LinearLayer(emb_dim, lab_dim, hid_dim=l1_dim, lr=lr, out_dim = class_num, 
                                    ae_cri="ce",ae_act=[nn.Tanh(),nn.Sigmoid()])
            else:
                layer = LinearLayer(emb_dim, lab_dim, hid_dim=l1_dim, lr=lr)
            layers.append(layer)
        self.layers = layers  
    
    
        
    def forward(self, x, y):
        
        layer_loss = []
        y = nn.functional.one_hot(y, self.class_num).float().to(y.device)
        
        x_side = self.sidedata(x)
        
        # forward function also update
        for idx,layer in enumerate(self.layers):
            
            # F Input shape [batch * seq_len * emb_dim] pool to [batch * 1 * emb_dim]
            # F Output shape [batch * emb_dim] == [batch * l1_dim]
            
            if self.same_emb:
                emb_side = self.emb(x_side[idx])
            else:
                emb_side = self.emb_layers[idx](x_side[idx])
                
            emb_side_pool = nn.functional.adaptive_max_pool2d(emb_side,(1,self.emb_dim))
            if idx == 0:             
                x_out, y_out, ae_out, as_out = layer(emb_side_pool.view(-1,self.emb_dim), y)
                layer_loss.append([ae_out.item(), as_out.item()])
            else:
                x_out = x_out.view(-1,1,self.emb_dim)
                x_cat = torch.cat((x_out, emb_side_pool), dim=1)
                x_cat = nn.functional.adaptive_max_pool2d(x_cat,(1,self.emb_dim))
                x_out, y_out, ae_out, as_out = layer(x_cat.view(-1,self.emb_dim), y_out)
                
                layer_loss.append([ae_out.item(), as_out.item()])
        return layer_loss
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        x_side = self.sidedata(x)
  
        for idx in range(len_path):
            if self.same_emb:
                emb_side = self.emb(x_side[idx].long())
            else:
                emb_side = self.emb_layers[idx](x_side[idx].long())
            emb_side_pool = nn.functional.adaptive_max_pool2d(emb_side,(1,self.emb_dim))
            
            if idx == 0:              
                x_out = self.layers[idx].enc.f(emb_side_pool.view(-1,self.emb_dim))        
                
            else:
                x_out = x_out.view(-1,1,self.emb_dim)
                x_cat = torch.cat((x_out, emb_side_pool), dim=1)
                x_cat = nn.functional.adaptive_max_pool2d(x_cat,(1,self.emb_dim))
                x_out = self.layers[idx].enc.f(x_cat.view(-1,self.emb_dim))                        
                     
        # bridge                    
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
        #print(self.layers[0].ae.h)    
        return y_out
    
class TransformerALsideText(alSideModel):    
    def __init__(self, vocab_size, num_layer, side_dim:List[int], emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None, same_emb=False):
        super().__init__(vocab_size, num_layer, side_dim, emb_dim, l1_dim, lr, class_num, lab_dim, word_vec, same_emb)

        
        # emb define
        if self.same_emb:
            if word_vec is not None:
                self.emb = nn.Embedding.from_pretrained(word_vec, freeze=False)
            else:
                self.emb = nn.Embedding(vocab_size, emb_dim) 
        else:
            emb_layers = ModuleList([])
            for idx in range(self.num_layer):
                if word_vec is not None:
                    emb = nn.Embedding.from_pretrained(word_vec, freeze=False)
                else:
                    emb = nn.Embedding(vocab_size, emb_dim) 
                emb_layers.append(emb)
            self.emb_layers = emb_layers
            
        # layer    
        layers = ModuleList([])
        for idx in range(self.num_layer):                
            if idx == 0:
                layer = TransLayer(emb_dim, lab_dim, l1_dim, lr=lr, out_dim = class_num,
                                   ae_cri="ce",ae_act=[nn.Tanh(),nn.Sigmoid()])
            else:
                layer = TransLayer(emb_dim, lab_dim, l1_dim, lr=lr)
            
            layers.append(layer)
        
        self.layers = layers  
           
    def forward(self, x, y):
        
        layer_loss = []
        mask = self.get_mask(x)
        y = nn.functional.one_hot(y, self.class_num).float().to(y.device)
        
        #print(f"x{x.shape}")
        x_side = self.sidedata(x)
        mask_side = self.sidedata(mask)
        
        # forward function also update
        for idx,layer in enumerate(self.layers):
            if self.same_emb:
                emb_side = self.emb(x_side[idx])
            else:
                emb_side = self.emb_layers[idx](x_side[idx])
                
            if idx == 0:
                mask_cat = mask_side[0]                
                x_out, y_out, ae_out, as_out, mask = layer(emb_side, y, mask_cat)
                layer_loss.append([ae_out.item(), as_out.item()])
            else:
                x_cat = torch.cat((x_out, emb_side), dim=1)
                mask_cat = torch.cat((mask_cat, mask_side[idx]), dim=1)
                
                x_out, y_out, ae_out, as_out, mask = layer(x_cat, y_out, mask_cat)
                layer_loss.append([ae_out.item(), as_out.item()])
        return layer_loss
    
    def get_mask(self, x):
        pad_mask = ~(x == 0)
        return pad_mask.cuda()
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        mask = self.get_mask(x)
        x_side = self.sidedata(x)
        mask_side = self.sidedata(mask)
        
        
        
        for idx in range(len_path):
            if self.same_emb:
                emb_side = self.emb(x_side[idx].long())
            else:
                emb_side = self.emb_layers[idx](x_side[idx].long())
            
            if idx == 0:
                mask_cat = mask_side[0]                
                x_out = self.layers[idx].enc.f(emb_side, mask_cat)        
                
            else:
                x_cat = torch.cat((x_out, emb_side), dim=1)
                mask_cat = torch.cat((mask_cat, mask_side[idx]), dim=1)
                x_out = self.layers[idx].enc.f(x_cat, mask_cat)                        
                     
        # bridge        
        denom = torch.sum(mask_cat, -1, keepdim=True)
        x_out = torch.sum(x_out * mask_cat.unsqueeze(-1), dim=1) / denom
            
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out
        
class LSTMALsideText(alSideModel):    
    def __init__(self, vocab_size, num_layer, side_dim:List[int], emb_dim, l1_dim, lr, class_num, lab_dim=128, word_vec=None, same_emb=False):
        super().__init__(vocab_size, num_layer, side_dim, emb_dim, l1_dim, lr, class_num, lab_dim, word_vec, same_emb)
        
        # emb 
        if self.same_emb:
            if word_vec is not None:
                self.emb = nn.Embedding.from_pretrained(word_vec, freeze=False)
            else:
                self.emb = nn.Embedding(vocab_size, emb_dim) 
            
        else:
            emb_layers = ModuleList([])
            for idx in range(self.num_layer):
                if word_vec is not None:
                    emb = nn.Embedding.from_pretrained(word_vec, freeze=False)
                else:
                    emb = nn.Embedding(vocab_size, emb_dim) 
                emb_layers.append(emb)
            self.emb_layers = emb_layers   
            
        # layer      
        layers = ModuleList([])
        for idx in range(self.num_layer):
              
            if idx == 0:
                layer = LSTMLayer(emb_dim, lab_dim, l1_dim, lr=lr, out_dim = class_num,
                                  ae_cri="ce",ae_act=[nn.Tanh(),nn.Sigmoid()])
            else:
                layer = LSTMLayer(emb_dim, lab_dim, l1_dim, lr=lr)
            
            layers.append(layer)
        self.layers = layers  
     
    def forward(self, x, y):
        
        layer_loss = []
        y = nn.functional.one_hot(y, self.class_num).float().to(y.device)
        
        x_side = self.sidedata(x)
        
        # forward function also update
        for idx,layer in enumerate(self.layers):
            
            # F Input shape [batch * seq_len * emb_dim]
            if self.same_emb:
                emb_side = self.emb(x_side[idx])
            else:
                emb_side = self.emb_layers[idx](x_side[idx])
            # emb_side_pool = nn.functional.adaptive_max_pool2d(emb_side,(1,self.emb_dim))
            if idx == 0:             
                x_out, y_out, ae_out, as_out, [h, _] = layer(emb_side, y)
                #x_out = torch.cat((x_out[:, :, :self.l1_dim], x_out[:, :, self.l1_dim:]), dim=-1)
                layer_loss.append([ae_out.item(), as_out.item()])
            else:
                x_cat = torch.cat((x_out[:, :, :self.l1_dim], x_out[:, :, self.l1_dim:]), dim=1)
                x_cat = torch.cat((x_cat, emb_side), dim=1)
                x_out, y_out, ae_out, as_out, [h, _] = layer(x_cat, y_out, h)

                layer_loss.append([ae_out.item(), as_out.item()])
                
        return layer_loss
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        x_side = self.sidedata(x)
  
        for idx in range(len_path):
            if self.same_emb:
                emb_side = self.emb(x_side[idx].long())
            else:
                emb_side = self.emb_layers[idx](x_side[idx].long())
            
            if idx == 0:              
                x_out, (h, c) = self.layers[idx].enc.f(emb_side)        
                
            else:
                #print(x_out.shape)
                x_cat = torch.cat((x_out[:, :, :self.l1_dim], x_out[:, :, self.l1_dim:]), dim=1)
                x_cat = torch.cat((x_cat, emb_side), dim=1)
                x_out, (h, c) = self.layers[idx].enc.f(x_cat, (h, c))                 
                     
        # bridge   
        h = h[0] + h[1]                 
        y_out = self.layers[len_path-1].enc.b(h)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out
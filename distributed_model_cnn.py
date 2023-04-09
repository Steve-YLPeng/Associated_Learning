from distributed_model import *

class CNNLayer(nn.Module):
    def __init__(self, conv:nn.Module, flatten_size:int, 
                 lab_dim:int, lr:float, out_dim:int=None,
                 ae_cri='mse', ae_act=[nn.Sigmoid(),nn.Sigmoid()] ):
        super().__init__()
        
        #TODO
        self.enc = ENC(None, conv=conv, out_dim=flatten_size, lab_dim=lab_dim, f='cnn',)
        if out_dim == None:
            self.ae = AE(lab_dim, lab_dim, cri=ae_cri, act=ae_act)
        else:
            self.ae = AE(out_dim, lab_dim, cri=ae_cri, act=ae_act)
        #
    
        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=lr)
        self.enc_opt = torch.optim.Adam(self.enc.parameters(), lr=lr)
        self.ae_sche = ReduceLROnPlateau(self.ae_opt, mode="max", factor=0.5, patience=3)
        self.enc_sche = ReduceLROnPlateau(self.enc_opt, mode="max", factor=0.5, patience=3)

    def forward(self, x, y):

        self.ae_opt.zero_grad()
        enc_y , ae_loss = self.ae(y)
        if self.training:
            ae_loss.backward()
            self.ae_opt.step()
    
        self.enc_opt.zero_grad()
        tgt = enc_y.detach()
        enc_x, enc_loss, _, _ = self.enc(x, tgt)
        if self.training:
            enc_loss.backward()
            self.enc_opt.step()

        return enc_x.detach(), enc_y.detach(), ae_loss, enc_loss

class CNN_AL(alModel):    
    def __init__(self, num_layer:int, l1_dim:int, lr:float, class_num:int, lab_dim:int=128):
        super().__init__(num_layer, l1_dim, class_num, lab_dim)
        
        self.num_layer = 4
        layers = ModuleList([])
        ### L0
        conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        block_1 = nn.Sequential(conv1, conv2)
        layer = CNNLayer(block_1, 32*32*32, lab_dim, out_dim=class_num, lr=lr,
                         ae_cri='ce', ae_act=[nn.Sigmoid(),nn.Sigmoid()] )
        layers.append(layer)
        ### L1
        conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        block_2 = nn.Sequential(conv3, conv4)
        layer = CNNLayer(block_2, 32*16*16, lab_dim, lr=lr,
                         ae_cri='mse', ae_act=[nn.Sigmoid(), None] )
        layers.append(layer)
        ### L2
        conv5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        block_3 = nn.Sequential(conv5, conv6)
        layer = CNNLayer(block_3, 64*16*16, lab_dim, lr=lr,
                         ae_cri='mse', ae_act=[nn.Sigmoid(), None] )
        layers.append(layer)
        ### L3
        conv7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias = True, padding = 1), nn.ReLU())
        conv8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 3, stride = 2, bias = True, padding = 1), nn.ReLU())
        block_4 = nn.Sequential(conv7, conv8)
        layer = CNNLayer(block_4, 64*8*8, lab_dim, lr=lr,
                         ae_cri='mse', ae_act=[nn.Sigmoid(), None] )
        layers.append(layer)
        
        self.layers = layers     
        
    def forward(self, x, y):
        layer_loss = []
        y = torch.nn.functional.one_hot(y, self.class_num).float().to(y.device)
        x_out = x
        y_out = y
        # forward function also update
        for idx,layer in enumerate(self.layers):
            x_out, y_out, ae_out, as_out = layer(x_out, y_out)
            layer_loss.append([ae_out.item(), as_out.item()])
        return layer_loss
    
    ### func for inference_adapt
    def layer_forward(self, x, idx):
        pass
    
    def bridge_return(self, x, len_path):
        pass
    
    @torch.no_grad()
    def inference_adapt(self, x, threshold=0.1, max_depth=None):
        #######################################################
        # x: batch of input sample
        # Samples with (entropy > threshold) will go to next layer
        # max_depth: the max depth of layer a sample will go throught
        #######################################################
        
        ###TODO
        return pred, entr
    
    @torch.no_grad()
    def inference(self, x, len_path=None):
        # full path inference by default
        if len_path == None:
            len_path = self.num_layer
            
        assert 1 <= len_path and len_path <= self.num_layer
        
        x_out = x
        for idx in range(len_path):
            x_out = self.layers[idx].enc.f(x_out)    
            
        y_out = self.layers[len_path-1].enc.b(x_out)
        
        for idx in reversed(range(len_path)):
            y_out = self.layers[idx].ae.h(y_out)
            
        return y_out


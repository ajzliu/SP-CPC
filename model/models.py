import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
        
# CNN Encoder Model
class Conv_Encoder(nn.Module):
    def __init__(self, kernel_size, embedding_dim, encoder_dim, sentence_len):
        super(Conv_Encoder, self).__init__()

        conv1d = nn.Conv1d(in_channels = embedding_dim, out_channels = encoder_dim, kernel_size = 1)
        # Section 3.3: sentence encoder genc (a 1D-convolution + ReLU + mean-pooling)
        self.conv_blocks = nn.Sequential(
            conv1d,
            nn.ReLU(),
            #nn.AvglPool1d(kernel_size = sentence_len)
        )

        '''
        AvgPool1d reduces the tensor from (64, 8, 7) 
        to (64, 120) instead of (64, 120, 8), so it might
        need to go
        '''
        
    def forward(self, x): # x is (B,S,D)   
        x = x.transpose(1,2)  # needs to convert x to (B,D,S)
        feature_extracted = self.conv_blocks(x) # feature_extracted is (B,D)
        
        return feature_extracted.squeeze()

# Contrastive Predictive Coding Model
class SCPC(nn.Module):
    def __init__(self, config, weights):
        super(SCPC, self).__init__()

        # load parameters
        self.enc_dimension = config.scpc_model.enc_dim
        self.ar_dimension = config.scpc_model.ar_dim
        self.k_size = config.scpc_model.k_size
        self.weights = weights

        # Calculate number of counties
        self.counties = config.dataset.counties

        # define type of encoder
        self.encoder = Conv_Encoder(
            kernel_size=1, 
            embedding_dim=config.scpc_model.emb_dim, 
            encoder_dim=config.scpc_model.enc_dim, 
            sentence_len=config.dataset.max_length
        )

        # define autoregressive model
        self.gru = nn.GRU(
            self.enc_dimension, 
            self.ar_dimension, 
            batch_first=True)
        
        # define predictive layer
        # self.Wk  = nn.ModuleList([nn.Linear(self.enc_dimension, self.ar_dimension) for i in range(self.k_size)])

        # Define predictive layer for every county
        self.Wk = nn.ModuleDict(
            {county: nn.ModuleList(
                [nn.Linear(self.enc_dimension, self.ar_dimension) for i in range(self.k_size)]
            ) for county in self.counties}
        )

    def init_hidden(self, batch_size, device = None):
        if device: return torch.zeros(1, batch_size, self.ar_dimension).to(device)
        else: return torch.zeros(1, batch_size, self.ar_dimension)

    def forward_MetropolisHastingsPrediction(self, c_t, x_t, ct_key, device):
        pred = torch.empty((self.k_size, 128, self.enc_dimension), dtype = torch.float, device=device) # pred (empty container) is (W2,B,D)
        # Tranformed optimal context vector c_t:
        for i in range(self.k_size):   # k_size=1
            linear = self.Wk[ct_key][i]
            # print(linear.shape)
            #print(c_t.shape)
            pred[i] = linear(c_t) # Wk*context is (B,D)
        # II. Metropolis-Hastings to predict x_tpk := x_{t+k}:
        Iter = 100   # 10000
        # Initial the future state x_tpk sampled from initial distribution N(0, Id_7) or x_tpk = torch.zeros(7,1):
        # mvn = dist.MultivariateNormal(torch.zeros(7), torch.eye(7))
        mvn = dist.MultivariateNormal(x_t, torch.eye(7, device=device))
        x_tpk = mvn.sample()
        x_tpk_save = torch.zeros(7,Iter, device=device)
        a_vals, r_vals = [], []
        # Iteration:
        for tau in range(Iter):
            mvn = dist.MultivariateNormal(x_tpk, torch.eye(7, device=device))
            x_tilde = mvn.sample()
            x_tilde = torch.reshape(x_tilde, (1,1,7))
            z_tilde = self.get_encoded(x_tilde)
            z_tilde = torch.reshape(z_tilde, (1,1,60))   # 120
            # z_tilde = z_tilde.view(128, 8, self.enc_dimension) # z_tilde is (B,W,D)
            x_tpk = torch.reshape(x_tpk, (1,1,7))
            z_tpk = self.get_encoded(x_tpk) # z_tpk is (B*W,D)
            z_tpk = torch.reshape(z_tpk, (1,1,60))   # 120
            # z_tpk = z_tpk.view(128, 8, self.enc_dimension) # z_tpk is (B,W,D)
            target = (z_tilde - z_tpk).transpose(0,1).to(device)
            hidden_size = 60   # 120
            logits = torch.matmul(
                pred.reshape([-1, hidden_size]),
                target.reshape([-1, hidden_size]).transpose(-1, -2),
            )
            # ratio:
            a = torch.exp( torch.tensor(logits) )
            a_vals.append(a[0])
            rho = torch.min( torch.tensor([ [a[0]], [1] ]) )
            r_vals.append(rho)
            # 2.4 Update a new sample with threshold gamma
            # gamma ~ Uni[0, 1]
            gamma = torch.rand(1)
            #
            if rho > gamma:
                x_tpk = x_tilde
            else:
                x_tpk = x_tpk
            # 2.5. save all MCMC samples:
            x_tpk_save[:,tau] = x_tpk
        # III. Output:
        predicted_state_k_save = x_tpk_save
       # print(predicted_state_k_save.shape)
        np_arr = predicted_state_k_save.cpu().numpy()
        # arr_to_plot = np_arr[0,:Iter]
        # predicted_state_k = x_tpk_save[:,Iter-1]
        predicted_state_k = np.mean(np_arr[:,50:], axis=1)
        return predicted_state_k, predicted_state_k_save, np_arr  #, a_vals, r_vals

    # B: Batch, W: Window, S: Sentence, D: Dimension
    def forward(self, x_dict): # x_dict = {x_1, ..., x_50}, x_j - 128 x 8 x 7
        # Separate the data into two arrays/tensors
        # 1. FIPS Codes aligned with data
        # 2. Data itself

        z_dict = {}
        context_dict = {}
        target_dict = {}
        pred_dict = {}
        for val in x_dict:
            val_tensor = x_dict[val]
            #val_tensor = val_tensor.type(torch.cuda.FloatTensor)
            batch, window, sentence_length = val_tensor.shape   # x_i is (B,W,S)
            device = val_tensor.device
            # create dummy hidden state
            hidden = self.init_hidden(batch, device) # hidden is (B,D)
            # Create a dictionary of all time series {x_jt}, i=1,...,50
            # x_dict = {x_1, ..., x_50}, x_j - 128 x 8 x 7
            # z_dict = {z_1, ..., z_50}, z_j - 128 x 8 x 120
            # get encoded values
            z_temp = self.get_encoded( val_tensor )                          # z is (B*W,D)
            z_dict[val] = z_temp.view(batch, window, self.enc_dimension)    # z is (B,W,D) = 128 x 8 x 120
            # separate forward and target samples
            # W1: forward window, W2: target window
            target_dict[val] = z_dict[val][:,-self.k_size:,:].transpose(0,1) # target is (W2,B,D)
            forward_sequence_i = z_dict[val][:,:-self.k_size,:] # forward_sequence is (B,W1,D)
            # feed ag model
            self.gru.flatten_parameters()
            output, hidden = self.gru(forward_sequence_i, hidden) # output is (B,W1,D)
            context_dict[val] = output[:,-1,:].view(batch, self.ar_dimension) # context is (B,D) (take last hidden state)
            pred_dict[val] = torch.empty((self.k_size, batch, self.enc_dimension), dtype = torch.float, device = device) # pred (empty container) is (W2,B,D)
            # loop of prediction

            for j in range(self.k_size):
                linear = self.Wk[val][j]
                pred_dict[val][j] = linear(context_dict[val]) # Wk*context is (B,D)

        # target is z, and pred is Wk*c_t
        loss, accuracy = self.info_nce_SCPCspatialloss_weightmatrix(pred_dict, target_dict)
        return loss, accuracy, context_dict, x_dict

    def info_nce_SCPCspatialloss_weightmatrix(self, prediction_dict, target_dict):
        I = len(target_dict)
        k_size, batch_size, hidden_size = target_dict[self.counties[0]].shape
        label = torch.arange(0, batch_size * k_size, dtype=torch.long, device=target_dict[self.counties[0]].device)
        # check dimensions of loss, accuracy in the old code!
        logits, loss, accuracy = {}, {}, {}
        #
        for county in target_dict:
            logits[county] = 0
            for other_county in target_dict:
                pred_calc = prediction_dict[other_county].reshape([-1, hidden_size])
                target_calc = target_dict[other_county].reshape([-1, hidden_size]).transpose(-1, -2)
                logits[county] = logits[county] + self.weights[county][other_county] * torch.matmul(pred_calc, target_calc)
            # Loss & accuracy of every time series:
            loss[county] = nn.functional.cross_entropy(logits[county], label, reduction='none')
            accuracy[county] = torch.eq(
                torch.argmax(F.softmax(logits[county], dim = 1), dim = 1),
                label)
        # total loss:
        total_loss = 0
        total_accuracy = 0
        for county in target_dict:
            total_loss = total_loss + loss[county]
            total_accuracy = total_accuracy + accuracy[county]
        # process for split loss and accuracy into k pieces (useful for logging)
        nce, acc = [], []
        for i in range(k_size):
            start = i * batch_size
            end = i * batch_size+batch_size
            nce.append(torch.sum(total_loss[start:end]) / batch_size)
            acc.append(torch.sum(total_accuracy[start:end], dtype=torch.float) / batch_size)
        return torch.stack(nce).unsqueeze(0), torch.stack(acc).unsqueeze(0)

    def get_encoded(self, x): # x is (B,S)
        z = self.encoder(x) # z is (B,D)
        return z
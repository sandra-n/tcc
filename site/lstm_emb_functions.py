import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from treat_tweets import treating_tweet
from joblib import load
import os

path = os.path.dirname(__file__)

device = torch.device('cpu')

class TweetsLSTM(nn.Module):
    def __init__(self,no_layers,hidden_dim,input_dim,drop_prob=0.5):
        super(TweetsLSTM,self).__init__()
 
        #self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
    
        #print(type(self.no_layers), type(self.input_dim), type(self.hidden_dim))
        
        #lstm
        self.lstm_layers = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        
        
        # dropout layer
        self.dropout_layer = nn.Dropout(p=drop_prob, inplace=True)
    
        # linear and sigmoid layer
        self.linear_layer = nn.Linear(self.hidden_dim, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self,x):
        batch_size, _seq1 , _seq2 = x.size()
        
        lstm_out, _ = self.lstm_layers(x)
        
        # dropout and fully connected layer
        out = self.dropout_layer(lstm_out.clone()) #(h_1) lstm_out

        lin_out = self.linear_layer(out)\
        
        # sigmoid function
        sig_out = self.sigmoid_layer(lin_out)
        #print("out2:", sig_out.shape)
        
        sig_out_f = sig_out[:,-1,:]
        # # return last sigmoid output and hidden state

        return sig_out_f

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros(self.no_layers,batch_size,self.hidden_dim).to(device)
        c0 = torch.zeros(self.no_layers,batch_size,self.hidden_dim).to(device)
        hidden = (h0,c0)
        return hidden

def load_lstm_emb_model():
    no_layers = 2
    input_dim = 1
    hidden_dim = 50 #256
    try:
        model = TweetsLSTM(no_layers,hidden_dim,input_dim,drop_prob=0.3)
        model.load_state_dict(torch.load(path + "/model/lstm_embedded/model_lstm_emb", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error("não foi possível encontrar um modelo para utilizar")

def tweets_tok(tweet):
    cntvct = load(path +'/model/lstm_embedded/cntvct.joblib')
    svd = load(path + '/model/lstm_embedded/svd.joblib')
    #st.info([tweet])
    tweet_cntvct = cntvct.transform(pd.Series(tweet))
    #st.info(tweet_cntvct)
    tweet_svd = svd.transform(tweet_cntvct)
    #st.info(tweet_svd[0])
    return np.array(tweet_svd)

def lstm_emb_pre_process_tweet(tweet_treated):
    tweet_tok = tweets_tok(tweet_treated)
    tweet_proc = torch.from_numpy(tweet_tok).type(torch.float32)
    tweet_proc = tweet_proc.unsqueeze(0)
    tweet_proc = torch.transpose(tweet_proc, 1, 2)
    
    return tweet_proc
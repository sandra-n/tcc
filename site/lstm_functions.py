import streamlit as st
import torch
import torch.nn as nn
import json
import numpy as np
from treat_tweets import treating_tweet
  

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
        
        h_1 = torch.zeros(self.no_layers, batch_size, self.hidden_dim)
        c_1 = torch.zeros(self.no_layers, batch_size, self.hidden_dim)
        
        hc_1 = (h_1, c_1)

        lstm_out, (h_1, c_1) = self.lstm_layers(x, hc_1)

        # dropout and fully connected layer
        out = self.dropout_layer(h_1) #(h_1) lstm_out

        lin_out = self.linear_layer(out)

        # sigmoid function
        sig_out = self.sigmoid_layer(lin_out)

        sig_out = sig_out[-1,:,:]
        
        return sig_out

def load_lstm_model():
    no_layers = 2
    input_dim = 1
    hidden_dim = 50 
    try:
        model = TweetsLSTM(no_layers,hidden_dim,input_dim,drop_prob=0.3)
        model.load_state_dict(torch.load("model/lstm/model_lstm_big", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error("não foi possível encontrar um modelo para utilizar")

def tweets_tok(tweet):
    file = open ('model/vocab_to_int.json', "r")
    vocab_to_int = json.load(file)

    tweet_tok = [[vocab_to_int[w]] for w in tweet.split() if w in vocab_to_int]
     
    return np.array(tweet_tok)

def padding(dataset_tok, max_len):
    for i in range(len(dataset_tok)):
        if len(dataset_tok[i]) < max_len:
            for z in range(max_len - len(dataset_tok[i])):
                dataset_tok[i] = [[0]] + dataset_tok[i]
    return np.array(dataset_tok)

def lstm_pre_process_tweet(tweet_treated):
    tweet_tok = tweets_tok(tweet_treated)
    tweet_padded = padding(tweet_tok, 280)
    tweet_proc = torch.from_numpy(tweet_padded).type(torch.float32)
    tweet_proc = tweet_proc.unsqueeze(0)
    return tweet_proc
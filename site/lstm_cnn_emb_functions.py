import streamlit as st
import torch
import torch.nn as nn
from joblib import load
import pandas as pd
import numpy as np
from treat_tweets import remove_stopwords, lower_tweet, splitPunctuation, removeLink, separateEmoji, removeMention

device = torch.device('cpu')

class TweetsLSTM(nn.Module):
    def __init__(self,no_layers,hidden_dim,input_dim,drop_prob=0.5):
        super(TweetsLSTM,self).__init__()
 
        #self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
    
        print(type(self.no_layers), type(self.input_dim), type(self.hidden_dim))
        
        # embedding and LSTM layers
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm
        self.lstm_layers = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        # CNN
        self.cnn_layer = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=5)
        self.maxpool_layer = nn.AdaptiveMaxPool1d(output_size=self.hidden_dim)
        
        # linear and sigmoid layer
        self.linear_layer = nn.Linear(self.hidden_dim, 1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self,x):
        #print("x:", x.shape, x.dtype)
        batch_size, _seq1 , _seq2 = x.size()
        
        lstm_out, _ = self.lstm_layers(x)
        #print("lstm out:",lstm_out.shape)
        
        out_cnn = self.cnn_layer(torch.reshape(lstm_out, (batch_size, self.hidden_dim, -1)))
        #print("cnn out: ", out_cnn)
        out = self.maxpool_layer(out_cnn)
        
        #print("out:", out.shape)
        lin_out = self.linear_layer(out)
        
        
        # sigmoid function
        sig_out = self.sigmoid_layer(lin_out)
        
        sig_out_f = sig_out[:,-1,:]
        
        return sig_out_f

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros(2*self.no_layers,batch_size,self.hidden_dim).to(device)
        c0 = torch.zeros(2*self.no_layers,batch_size,self.hidden_dim).to(device)
        hidden = (h0,c0)
        return hidden

def load_lstm_cnn_emb_model():
    no_layers = 2
    input_dim = 1
    hidden_dim = 50 
    try:
        model = TweetsLSTM(no_layers,hidden_dim,input_dim,drop_prob=0.3).to(device)
        model.load_state_dict(torch.load("model/lstm_cnn_embedded/model_lstm_cnn_emb", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error("não foi possível encontrar um modelo para utilizar")

def tweets_tok(tweet):
    cntvct = load('./model/lstm_cnn_embedded/cntvct.joblib')
    svd = load('./model/lstm_cnn_embedded/svd.joblib')
    tweet_cntvct = cntvct.transform(pd.Series(tweet))
    tweet_svd = svd.transform(tweet_cntvct)
    return np.array(tweet_svd)

def lstm_cnn_emb_pre_process_tweet(tweet):
    tweet_treated = lower_tweet(remove_stopwords(removeMention(removeLink(splitPunctuation(separateEmoji(tweet))))))
    tweet_tok = tweets_tok(tweet_treated)
    tweet_proc = torch.from_numpy(np.array(tweet_tok)).type(torch.float32)
    tweet_proc = tweet_proc.unsqueeze(0)
    tweet_proc = torch.transpose(tweet_proc, 1, 2)
    return tweet_proc
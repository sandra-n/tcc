import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from treat_tweets import treating_tweet
import pickle
from keras_preprocessing.sequence import pad_sequences
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
    
        # print(type(self.no_layers), type(self.input_dim), type(self.hidden_dim))
        
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
        # print("x:", x.shape, x.dtype)
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
        
        sig_out = sig_out[:,-1,:]
        #print(sig_out.shape)
        return sig_out

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros(self.no_layers,batch_size,self.hidden_dim).to(device)
        c0 = torch.zeros(self.no_layers,batch_size,self.hidden_dim).to(device)
        hidden = (h0,c0)
        return hidden

def load_lstm_cnn_model():
    no_layers = 2
    input_dim = 1
    hidden_dim = 50 
    try:
        model = TweetsLSTM(no_layers,hidden_dim,input_dim,drop_prob=0.3)
        model.load_state_dict(torch.load(path + "/model/lstm_cnn/model_lstm_cnn", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error("não foi possível encontrar um modelo para utilizar")

def lstm_cnn_pre_process_tweet(tweet_treated):
    max_length = 280
    file = open(path + "/model/cnn/tokenizer.pickle",'rb')
    tokenizer = pickle.load(file)
    tweet_tokenized = tokenizer.texts_to_sequences([tweet_treated])
    tweet_padded = pad_sequences(tweet_tokenized, maxlen=max_length, padding='post')
    tweet_proc = torch.from_numpy(np.array(tweet_padded)).type(torch.float32)
    tweet_proc = tweet_proc.unsqueeze(0)
    tweet_proc = torch.transpose(tweet_proc, 1, 2)
    return tweet_proc
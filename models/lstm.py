import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

def tweets_input(dataset, n):
    all_tweets = []
    for tweet in dataset['text']:
        words = tweet.split()
        all_tweets.append(words)

    counter_words = Counter(all_tweets)
    counter_words = counter_words.most_common(n)

    return counter_words

class TweetLSTM(nn.Module):
    def __init__(self, n_input, n_hidden):
        self.input_dim = n_input
        self.hidden_dim = n_hidden

        self.emb_layer = nn.Embedding()
        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_dim, 3)
        self.softmax_layer = nn.Softmax(3)

    def forward(self):



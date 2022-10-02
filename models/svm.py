from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer   
from keras.utils.data_utils import pad_sequences
from sklearn import model_selection, svm
from sklearn import metrics
import pandas as pd
from treat_tweets import *
from pathlib import Path


def get_dataset():
    cwd = Path.cwd()
    path_covid_labelled = "datasets/covid_hate/annotated_tweets_w_text.csv"
    relative_path_covid_labelled = Path(cwd.parent, path_covid_labelled)

    #covid labeled
    covid_labeled = pd.read_csv(relative_path_covid_labelled)
    covid_labeled = covid_labeled.drop_duplicates()
    covid_labeled = covid_labeled.rename(columns={'Tweet ID':'id','Text':'text','BERT_label':'label'})
    covid_labeled = covid_labeled.drop(covid_labeled[covid_labeled['label'] == 0].sample(n=1000).index)

    #asian racism terms labeled
    path_asian_labelled = "datasets/labeled_asian_for_test.csv"
    relative_path_asian_labelled = Path(cwd.parent, path_asian_labelled)
    asian_labeled = pd.read_csv(relative_path_asian_labelled)
    asian_labeled = asian_labeled.drop(asian_labeled[asian_labeled['label'] == 0].sample(n=700).index)

    #rest racism terms labeled
    path_rest_labelled = "datasets/labeled_rest_for_test.csv"
    relative_path_rest_labelled = Path(cwd.parent, path_rest_labelled)
    rest_labeled = pd.read_csv(relative_path_rest_labelled)
    rest_labeled = rest_labeled.drop(rest_labeled[rest_labeled['label'] == 0].sample(n=400).index)

    #concat covid labeled, rest and asian labeled tweets
    train_dataset = pd.concat([asian_labeled,rest_labeled],ignore_index=True)
    train_dataset = train_dataset.dropna(subset=['label']).drop(columns=['Unnamed: 0','time','author_id','attachments'])
    train_dataset = pd.concat([train_dataset,covid_labeled])

    path_covid_test = "datasets/covid_hate/covid_long_dataset_labeled.feather"
    relative_path_covid_test = Path(cwd.parent, path_covid_test)
    covid_test = pd.read_feather(relative_path_covid_test)

    return train_dataset, covid_test

def one_hot_pre_processing(train_x, test_x, vocab_size): #
    train_x_one_hot = [one_hot(d, vocab_size, filters='', lower=True, split=' ') for d in train_x]
    test_x_one_hot = [one_hot(d, vocab_size, filters='', lower=True, split=' ') for d in test_x]
    return train_x_one_hot, test_x_one_hot

def tokenizer_pre_processing(train_x, test_x):    
    tokenizer = Tokenizer(num_words=500)
    tokenizer.fit_on_texts(train_x)
    train_x_tokenized = tokenizer.texts_to_sequences(train_x)
    tokenizer.fit_on_texts(test_x)
    test_x_tokenized = tokenizer.texts_to_sequences(test_x)
    return train_x_tokenized, test_x_tokenized

def pad_sentences(train_x, test_x):
    max_length = 100
    train_x_padded = pad_sequences(train_x, maxlen=max_length, padding='post')
    test_x_padded = pad_sequences(test_x, maxlen=max_length, padding='post')
    return train_x_padded, test_x_padded

def svm_classifier(train_x, train_y, test_x, test_y):
    clf = svm.SVC(kernel='linear') 
    clf.fit(train_x, train_y)

    y_pred = clf.predict(test_x)
    print("Accuracy:",metrics.accuracy_score(test_y, y_pred))

    
print("getting datasets...")
train_dataset, covid_test = get_dataset()
train_dataset['label'] = train_dataset['label'].replace({'1': '0'})
train_dataset['label'] = train_dataset['label'].replace({'2': '1'})

covid_test = covid_test.rename(columns={'BERT_label':'label'})

train_dataset = pd.concat([train_dataset,covid_test],ignore_index=True)

print("creating each test and train sets...")
train_x, test_x, train_y, test_y = model_selection.train_test_split(train_dataset['text'],train_dataset['label'],test_size=0.3)
#test_x = test_dataset['text']
#test_y = test_dataset['label']

#create train specific arrays
#train_x = train_dataset['text']
#train_y = train_dataset['label']
#train_ids = train_dataset['id'].tolist()

train_x = train_x.apply(lambda x:removeMention(x)).apply(lambda x:removeLink(x))

#print("one hot pre processing...")
#train_x, test_x = one_hot_pre_processing(train_x, test_x, 500)
print("tokenizing...")
train_x, test_x = tokenizer_pre_processing(train_x, test_x)
print("padding...")
train_x, test_x = pad_sentences(train_x, test_x)
print("using the classifier...")
svm_classifier(train_x, train_y, test_x, test_y)
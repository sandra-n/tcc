import pandas as pd
import numpy as np
import glob
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from treat_tweets import *
from lib import *
from pathlib import Path

def read_files_csv(path):
    print("start reading...")
    complete_path = path + "*.csv"
    print(complete_path)
    csv_files = glob.glob(complete_path)
    df = []
    counter = 1
    for f in csv_files:
        csv = pd.read_csv(f, header=None, names=[0,1,2,3,4,5])
        df.append(csv)
        print("files appended: " + str(counter))
        counter = counter + 1
    print("concatenating...")
    df = pd.concat(df)
    df = df.drop_duplicates(subset=[1]).dropna(subset=[2])
    print(df)
    return df

#rest = read_files_csv("C:/Users/Gabriel Sanefuji/Desktop/tcc/tcc/datasets/rest/")
#asian = read_files_csv("C:/Users/Gabriel Sanefuji/Desktop/tcc/tcc/datasets/asian/")
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

def get_pre_processed_dataset():
    cwd = Path.cwd()
    path_train_dataset = "datasets/train_dataset.csv"
    relative_path_train_dataset = Path(cwd.parent, path_train_dataset)
    path_test_dataset = "datasets/test_dataset.csv"
    relative_path_test_dataset = Path(cwd.parent, path_test_dataset)
    train_dataset = pd.read_csv(relative_path_train_dataset)
    test_dataset = pd.read_csv(relative_path_test_dataset)
    return train_dataset, test_dataset

def pre_process(dataset):
    # Step - a : Remove blank rows if any.
    dataset['text'].dropna(inplace=True)# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    dataset['text'] = [entry.lower() for entry in dataset['text']]# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    dataset['text']= [word_tokenize(entry) for entry in dataset['text']]# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(dataset['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        dataset.loc[index,'text_final'] = str(Final_words)
    dataset['text_final'].dropna(inplace=True)
    return dataset

print("getting datasets...")
train_dataset, test_dataset = get_dataset()
train_dataset['label'] = train_dataset['label'].replace({'1': '0'})
train_dataset['label'] = train_dataset['label'].replace({'2': '1'})

test_dataset = test_dataset.rename(columns={'BERT_label':'label'})

print("preprocessing datasets...")
train_dataset = pre_process(train_dataset)
test_dataset = pre_process(test_dataset)

#train_dataset.to_csv(r"path/to/new/file")
#test_dataset.to_csv(r"path/to/new/file")

print("creating each test and train sets...")
test_x = test_dataset['text_final']
test_y = test_dataset['label']

#create train specific arrays
train_x = train_dataset['text_final']
train_y = train_dataset['label']
train_ids = train_dataset['id'].tolist()

#train_x_data_prepared = train_x.apply(lambda x:removeMention(x)).apply(lambda x:removeLink(x))

print("enconding...")
Encoder = LabelEncoder()
train_y = Encoder.fit_transform(train_y)
test_y = Encoder.fit_transform(test_y)

print("using tfidf...")
tfidf_vect_train = TfidfVectorizer(max_features=7000)
tfidf_vect_train.fit(train_dataset['text_final'].apply(lambda x: np.str_(x)))
train_x_tfidf = tfidf_vect_train.transform(train_x.apply(lambda x: np.str_(x)))

tfidf_vect_test = TfidfVectorizer(max_features=7000)
tfidf_vect_test.fit(test_dataset['text_final'].apply(lambda x: np.str_(x)))
test_x_tfidf = tfidf_vect_test.transform(test_x.apply(lambda x: np.str_(x)))


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
print("using the classifier...")
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_x_tfidf, train_y)# predict the labels on validation dataset
predictions_SVM = SVM.predict(test_x_tfidf)# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, test_y)*100)
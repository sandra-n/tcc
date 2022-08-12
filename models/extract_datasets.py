from lib import *

covid_long_dataset = pd.read_feather()
covid_short_dataset = pd.read_feather()
asian_rest_dataset = pd.read_feather()

train_dataset = asian_rest_dataset.dropna(subset=['label']).drop(columns=['Unnamed: 0', 'time', 'author_id', 'attachments'])
train_dataset = pd.concat([train_dataset, covid_short_dataset])

train_y = train_dataset['label']
train_x = train_dataset['text']
train_ids = train_dataset['id'].tolist()

test_dataset = covid_long_dataset.rename(columns={1:'id',2:'text',3:'time',4:'author_id',5:'attachments'})
test_dataset['id'] = test_dataset['id'].astype(str)
test_dataset = test_dataset.drop(columns=[0,'time','author_id','attachments'])
test_dataset = test_dataset.drop_duplicates()
test_dataset = test_dataset[~test_dataset['id'].isin(train_ids)]
test_dataset
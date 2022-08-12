import numpy as np
import pandas as pd
from emoji import UNICODE_EMOJI
import re
import nltk
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words('english'))
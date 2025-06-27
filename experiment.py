#1- data collection
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import pandas as pandas

#load the dataset
data=gutenberg.raw('shakespeare-hamlet.txt')

## save to a file
with open('hamlet.txt','w') as file:
  file.write(data)


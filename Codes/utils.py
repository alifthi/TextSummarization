import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
class utils():
    def __init__(self,dataAddr):
        self.dataAddr = dataAddr
        self.trainData = self.loadData()
        self.testData = self.loadData('test')
        self.validationData = self.loadData('validation')
    def loadData(self,type='train'):
        Data = pd.read_csv(self.dataAddr+'/'+type+'.csv')
        return Data
    def describeTrainData(self):
        count = self.trainData['article'].str.count(' ')
        print('Data frame columnts:\n',self.trainData.columns)
        count.describe()
    @staticmethod
    def preprocessing(data):
        data['article'] = data['article'].str.replace('[^\w\s]','')
        data['article'] = data['article'].str.replace('\([^)]*\)','')
        data['highlights'] ='_start_' + ' ' +data['highlights'].str.replace('[^\w\s]','')+ ' ' +'_end_' 
        tok = tf.keras.preprocessing.text.Tokenizer()
        tok.fit_on_texts(list(data['article'].astype(str))) 
        text = tok.texts_to_sequences(list(data['article'].astype(str)))
        text = tf.keras.preprocessing.sequence.pad_sequences(text)
        tok = tf.keras.preprocessing.text.Tokenizer() 
        tok.fit_on_texts(list(data['highlights'].astype(str)))
        summary= tok.texts_to_sequences(list(data['highlights'].astype(str)))
        summary = tf.keras.preprocessing.sequence.pad_sequences(summary)
        return [text,summary]
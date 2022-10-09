from json import decoder
import tensorflow as tf
import numpy as np
from utils import utils


class Model():
    def __init__(self,inputLength,latentDim,encoderVocabSize,decoderVocabSize): 
        self.inputLengh = inputLength
        self.latentDim = latentDim
        self.encoderVocabSize = encoderVocabSize
        self.decoderVocabSize = decoderVocabSize
        self.net = self.buildModel()
    def buildModel(self):
        encoderInput = tf.keras.Input(shape=[self.inputLengh,])
        x = tf.keras.layers.Embedding(self.encoderVocabSize,
                                      self.latentDim,
                                      mask_zero=False)(encoderInput)
        x = tf.keras.layers.BatchNormalization()(x)
        _,state = tf.keras.layers.GRU(self.latentDim,return_state = True)(x)
        encoder = tf.keras.Model(inputs = encoderInput,outputs = state)
        encoderOutput = encoder(encoderInput)
        decoderInput = tf.keras.Input(shape=[None,])
        x = tf.keras.layers.Embedding(self.decoderVocabSize,self.latentDim,mask_zero=False)(decoderInput)
        x = tf.keras.layers.BatchNormalization()(x)
        x,_ = tf.keras.layers.GRU(self.latentDim,return_state = True,
                                  return_sequences = True)(x,initial_state = encoderOutput)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(self.decoderVocabSize,activation='softmax')(x)
        model = tf.keras.Model([encoderInput,decoderInput],x)
        return model
    def compileModel(self):
        self.net.summary()
        self.net.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                         loss = tf.keras.metrics.sparse_categorical_crossentropy)
    def train(self,encoderInput,decoderInput,decoderTarget):
        self.net.fit([encoderInput,decoderInput],decoderTarget,epochs = 1,batch_size=16)
    def saveModdel(self,addr='./Model'):
        model.save(self.net,addr+'/modelstruct.h5')
        model.saveweights(self.net,addr+'/modelweights.h5')
utils = utils('/home/alifathi/Documents/AI/DATA/NLP/cnn_dailymail')
utils.describeTrainData()
[corpus,summary,encoderVocabSize,decoderVocabSize] = utils.preprocessing(utils.trainData)
decoderInputData = summary[:,:-1]
decoderTarget = summary[:,1:]
model = Model(inputLength=corpus.shape[1],latentDim=100,
              encoderVocabSize = encoderVocabSize,
              decoderVocabSize=decoderVocabSize)
model.compileModel()
model.train(encoderInput=corpus,decoderInput=decoderInputData,decoderTarget=decoderTarget)
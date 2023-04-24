import tensorflow as tf
class Model():
    def __init__(self,inputLength,latentDim,encoderVocabSize,decoderVocabSize): 
        self.inputLengh = inputLength
        self.latentDim = latentDim
        self.encoderVocabSize = encoderVocabSize
        self.decoderVocabSize = decoderVocabSize
        self.net = self.buildModel()
    def buildModel(self):
        encoderInput = tf.keras.Input(shape=[self.inputLengh])
        x = tf.keras.layers.Embedding(self.encoderVocabSize,
                                      self.latentDim,
                                      mask_zero=False)(encoderInput)
        x = tf.keras.layers.BatchNormalization()(x)
        _,state = tf.keras.layers.GRU(self.latentDim,return_state = True)(x)
        encoder = tf.keras.Model(inputs = encoderInput,outputs = state)
        encoderOutput = encoder(encoderInput)
        decoderInput = tf.keras.Input(shape=[None])
        x = tf.keras.layers.Embedding(self.decoderVocabSize,self.latentDim,mask_zero=False)(decoderInput)
        x = tf.keras.layers.BatchNormalization()(x)
        x,_ = tf.keras.layers.GRU(self.latentDim,return_state = True,
                                  return_sequences = True)(x,initial_state = encoderOutput)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(self.decoderVocabSize,activation='softmax')(x)
        model = tf.keras.Model([encoderInput,decoderInput],x)
        return model
    def compileModel(self,opt = tf.keras.optimizers.Adam(learning_rate = 0.001),loss = tf.keras.metrics.sparse_categorical_crossentropy):
        self.net.summary()
        self.net.compile(optimizer = opt,loss = loss)
    def train(self,encoderInput,decoderInput,decoderTarget,epochs = 1,batchSize = 16):
        self.net.fit([encoderInput,decoderInput],decoderTarget,epochs = epochs,batch_size=batchSize)
    def saveModdel(self,addr='./Model',mode = 'Model'):
        if mode == 'Model':
            self.net.save(self.net,addr+'/modelstruct.h5')
        elif mode == 'weights':
            self.net.saveweights(self.net,addr+'/modelweights.h5')
    def loadModel(self,addr):
        self.net = tf.keras.models.load_model(addr)
        
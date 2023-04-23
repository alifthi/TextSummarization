from utils import utils
from Model import Model
dataDir = r"C:\Users\alifa\Documents\AI\DATA\NLP\cnn_dailymail\\"
util = utils(dataDir)
util.loadData()
text,decoderInput,decoderOutput,decoderVocabSize,encoderVocabSize = util.preprocess(util.trainData)


model = Model(inputLength=text.shape[1],latentDim=100,
              encoderVocabSize = encoderVocabSize,
              decoderVocabSize=decoderVocabSize)
model.compileModel()
model.train(encoderInput=text,decoderInput=decoderInput,decoderTarget=decoderOutput)
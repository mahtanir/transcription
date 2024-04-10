#Can convert from raw audio to mel spectogram (i.e frequency vs time and color coded)

import torchaudio 
from characters import *
from torch import nn
from torch.nn import functional as F
import torch

train_dataset = torchaudio.datasets.LIBRISPEECH('./', url='train-clean-100', download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH('./', url="test-clean", download=True)
#waveform, sample_rate, and transcript + additional metadata speaker, chapter, utteranceid 

#Data augmentation -> changing pitch, speed, reverb / vibrato?, noise 
#As an alternative, can also cut out consecutive time and frequency chunks (found to result in a model having stronger 
#generalisation capabolities i.e generalise to test data -> SPEC AUGMENT) //particularly for speech recognition. 


class TextTransform:
    def __init__(self) -> None:
        self.charToIndex = {} 
        self.indexToChar = {} 
        self.characters = char_map_str()
        lines = self.characters.char_map_str.strip().split()
        for line in lines: 
            ch, index = line.split() 
            if (ch == '<SPACE>'):
                self.charToIndex[' '] = int(index)
            else:
                self.charToIndex[ch] = int(index)
            self.indexToChar[int(index)] = ch
        self.indexToChar[1] = " "
    
    def text_to_int(self, text):
        int_sequence = [] 
        for char in text: 
            int_sequence.append(self.charToIndex[char])
        return int_sequence 
    
    def int_to_text(self, int_sequence):
        char = ''
        for val in int_sequence:
            char += self.indexToChar[val]
        return char


train_audio_transforms = nn.Sequential( #like a pipe operator. 
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128), #frequencies (y -axis) for a given time window (x-axis)
    # amplitude is the color in decibles. Mel is a math transofrmation to the frquency such that equidistant frequencies
    # also sound equidistant to the human ear. This is the n_mels i.e how many mel filterbanks to use (filterbanks = a range of freuqencies).
    #Not equidistantly spaced frequency buckets like normal spectorgrams but rather more logarithmic like.  
    #Mel spectogram is used to provide sound information similar to what humans would perceive. 
    # Sample rate is used to compute max frequency using nyquist theorem (i.e sample rate / 2). Then seaparate to the bins/ filters. 
    #Often times convolution + Mel Spectogram seen as a better alternative to audio signal + RNN 
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    #f consecutive mel frequency channels are masked [f0, f0 + f] 
    # f is first chosen from a uniform distribution from 0 to the frequency mask parameter F, and f0 is chosen from 0, ν − f
    #masked horizontally. kinda like the width. 
    torchaudio.transforms.TimeMasking(time_mask_param=35)
    # https://chat.openai.com/c/e0574e99-e988-42f9-be07-c94a1cb0072f (size is n_mels, time)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram() 
#default params all g --> can pass torchaudio.load('.wav') where waveform output will be tensor array that when normalised can be passed
text_transform = TextTransform()

def data_preprocessing(data, data_type="train"): #within the batch, we take all the data within it, and then we process it as follows. 
    #i.e we create arrays so they're easy to unpack
    spectograms = []
    labels = []
    input_lengths= [] 
    label_lengths = [] 

    for (waveform, _, transcript, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0,1)
        else:
            spec = valid_audio_transforms(waveform).transpose(0,1) #switch to time, n_mels (we do this for pad sequence later which
            #ensures that all sepctograms are of the same time dimension (i.e the longest time dimension input in the data)
        spectograms.append(spec)
        label = text_transform.text_to_int(transcript.lower())
        labels.append(label) #list of torches of int attays 
        input_lengths.append(spec.shape[0]//2) #time / 2 WHY not just shape[0]???
        label_lengths.append(len(label))


#spectogram helps to quantise!!! i.e not continuos frequencies data. aggregate amplitude on the equidistant frequency buckets or filters here. 
        #also aligned with human representation? 
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3) 
    #Shape: Batch size, 1, n_mels, time   ((it's 2d) for each frequency index and time we have amplitude values, 1 for channel makes it easier for pytorch to interpret)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True) #Batch size (total count), Max length, 1 ??/ may not have the 1. 
    print(labels.shape)
    return spectrograms, labels, input_lengths, label_lengths


class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats) -> None:
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats) #n_feats refers to the dimension of features. It will search for last dim andnormalise over that dimension, i.e features, where it's expected that the last dimension will have size of n_feat. 

    def forward(self, x):
        #x is (batch, channels, features, time)
        x = x.transpose(2,3).contiguos() #continguos reformates memory storage and converting to have features at the last dim 
        x = self.layer_norm(x) 
        x = x.transpose(2,3).contiguos() 
        #makes sense, i.e care moreso relative to each other in a timestemp vs raw amplitude. 

#maybe convolutions are like learning speech chunk representations and harmonies. The greater reference window can 
        #result in more accurate transcriptions because it has some context on future. Also just learns what 
        #sounds correspond to which spellings 

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, n_feats, dropout) -> None:
        super(ResidualCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, padding=(kernel // 2), stride=stride) #will always be same size input and output
        self.cnn2 = nn.Conv2d(in_channels, out_channels, kernel, padding=(kernel // 2), stride=stride)
        self.layernorm1 = CNNLayerNorm(n_feats)
        self.layernorm2 = CNNLayerNorm(n_feats)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x #(batch, channel, feature, time) Actually, feature here very much so just like frequency
        x = self.layernorm1(x) #we use a cnn first prior to reisdual - for each timestep normalise over frequency amplitude vals.
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x) 
        x = self.layernorm2(x) #we use a cnn first prior to reisdual 
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.cnn2(x) 
        return x + residual

class BidrectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batchFirst) -> None:
        super(BidrectionalLSTM).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=batchFirst)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = F.relu(x)
        #not sure why no dropout here 
        x, _, _ =  self.lstm(x) #x is output from last of stacked LSTM num_layers where output is hidden state (not softmax yet)
        x = self.dropout(x) 
        return x 

class SpeechRecognitionModel(nn.Module):
    def __init__(self, lstm_input_dim, hidden_size, n_cnn_layers, n_rnn_layers,  n_class, resChannels=32, kernel=3, stride=2, n_feats=128, num_layers=1, dropout=0.1) -> None:
        super().__init__()
        n_feats = n_feats // 2 #I think this is because due to padding of 1, stride of 2, and f = 3, then (n - 3 + 2) / 2 ~ n/2  
        #cnn first detects edges i.e harmonies, then can even learn things like rhythm and structure of audio. 
        self.cnn = nn.Conv2d(1, 32, 3, stride, padding = 3//2) 
        self.residualConvs = nn.ModuleList()
        for i in range(n_cnn_layers):
            self.residualConvs.append(ResidualCNN(resChannels, resChannels, kernel, 1, n_feats, dropout))
        self.linear = nn.Linear(32*n_feats, lstm_input_dim) #this represents leftover frequencies and the 32 channels, that's the data per timestep 
        self.lstms = nn.ModuleList() 
        for i in range(n_rnn_layers):
            self.lstms.append(BidrectionalLSTM(input_size=lstm_input_dim if i == 0 else hidden_size * 2, 
                                               hidden_size=hidden_size, num_layers=1, dropout=dropout, batchFirst= True)) #why i==0 here 
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_class)
        )
    
    def forward(self, x): 
        x = self.cnn(x)
        #maybe unlikely given sampling rate for orignal cnn reduce time dimension to make a difference? Size becomes around n/2 
#and adds context, but it really doesn't matter much because sampling rate, i.e samples per second is 16000??
        #yup I think this is correct, see https://kouohhashi.medium.com/dissecting-deepspeech-pytorch-part-2-c029042b30b0
        #note conv is not casual but forward looking here. 
        x = self.residualConvs(x)
        #need to prepare to flatten i.e linear will affect the last layer only but need to make it of the form 
        #(N, time, channels*n_freq)
        # x right now is (Batch, channels, frequency, time)
        # x = x.transpose(2,3)
        # x = x.view(x.shape[0], x.shape[2], x.shape[1]*x.shape[3])
        x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) #like a flatten
        x = x.transpose(1,2) #i.e now we have channels * frquency as last one for linear layer to process
        x = self.linear(x)
        x = self.lstms(x)
        return self.classifier(x)
        

#how come we don't add input length here?
def greedyDecoder(output, labels, label_lengths, blank_labels=28, collapse_repeated=True): #i.e max char prob from softmax for each timestep (doesn't account for marginalisation). 
#The concept of text generation with LSTMS and greedy not being accurate cause next output conditional on previous 
#doesn't apply here I feel because the input audio is predetermined. I.e not generative here! 
    arg_maxes = torch.argmax(output, dim=2) #batch, time, n_classes  
    decodes = [] #now argmaxes is batch, time 
    targets = [] 
    for i, prob_matrix in enumerate(arg_maxes): #i.e we are iterating through the batch which is really just one sample transcipt
        target = labels[i][:label_lengths[i]].tolist() #latter slicing because of the padding 0s to end due to max label_length 
        targets.append(text_transform.indexToChar(target))
        decode = [] 
        for j, timestep_output in enumerate(prob_matrix):
            if (timestep_output != blank_labels): #i.e if epsilon then we don't want to append BUT in subsequent will ensure we don't collapse 
                if (collapse_repeated and j > 0 and prob_matrix[j] == prob_matrix[j-1]):
                    continue 
                decode.append(timestep_output)
        decodes.append(text_transform.indexToChar(timestep_output))
    return decodes, targets
        

    



#SGD takes minibatches the same as ADAM. The difference is that ADAM adjusts learning rates for parameters separately while SGD does them together.
# That allows ADAM to converge fast since one learning rate is unlikely to be best for all parameters in a model;
    
# The One Cycle Learning Rate Scheduler was first introduced in the paper Super-Convergence:
# Very Fast Training of Neural Networks Using Large Learning Rates. 
# This paper shows that you can train neural networks an order of magnitude faster, 
# while keeping their generalizable abilities, using a simple trick. You start with a low learning rate, 
# which warms up to a large maximum learning rate, then decays linearly to the same point of where you originally started.
    #Perhaps also regularisation benefits because learning rate at max is >> than min learning rate so severely penalises params? 
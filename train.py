import torch
from torch import nn  
from transcription.transcription import * 
from torch.nn import functional as F 
from scorer import *


class IterMeter(object):
    def __init__(self) -> None:
        self.val = 0 
    def increment(self):
        self.val += 1 
    def get(self):
        return self.val 

def load_model(hparams):
    model =  SpeechRecognitionModel(hparams['lstm_input_dim'], hidden_size=hparams['hidden_size'], 
                                   n_cnn_layers=hparams['n_cnn_layers'], n_rnn_layers=hparams['n_rnn_layers'],
                                   n_class=hparams['n_class'])
    model.load_state_dict(torch.load('weights/weights.pth'))
    return model 

def test_final(hparams, trainloader, testloader):
    model = load_model(hparams=hparams)
    loss_crit = nn.CTCLoss(blank=28)
    test_cer = [] 
    test_wer = [] 
    for batch_ix, batch in enumerate(testloader):
        spectograms, labels, input_lengths, label_lengths = batch
        output = model(spectograms) #N here = length of the spectogram 
        output = F.softmax(output, dim=2) #softmax of N, T, n_channels 
        output = output.transpose(0,1)
        loss = loss_crit(output, labels, input_lengths, label_lengths)
        agg_loss = agg_loss + loss.item() / len(testloader)
        decoded_output, target_output = greedyDecoder(output.transpose(0,1), labels, label_lengths)
        for j in range(len(decoded_output)):
            test_cer.append(cer(target_output[j], decoded_output[j]))
            test_wer.append(wer(target_output[j], decoded_output[j]))
    
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    print(f'word error rate: {avg_wer} vs. average character error rate: {avg_cer}\n')
        

def test_incremental(model, testloader, epoch):
    with torch.no_grad():
        loss_crit = nn.CTCLoss(blank=28)
        test_cer = [] 
        test_wer = [] 
        for batch_ix, batch in enumerate(testloader):
            spectograms, labels, input_lengths, label_lengths = batch
            output = model(spectograms) #N here = length of the spectogram 
            output = F.softmax(output, dim=2) #softmax of N, T, n_channels 
            output = output.transpose(0,1)
            loss = loss_crit(output, labels, input_lengths, label_lengths)
            agg_loss = agg_loss + loss.item() / len(testloader)
            decoded_output, target_output = greedyDecoder(output.transpose(0,1), labels, label_lengths)
            for j in range(len(decoded_output)):
                test_cer.append(cer(target_output[j], decoded_output[j]))
                test_wer.append(wer(target_output[j], decoded_output[j]))
        
        avg_cer = sum(test_cer)/len(test_cer)
        avg_wer = sum(test_wer)/len(test_wer)
        print(f'FOR EPOCH: {epoch}, word error rate: {avg_wer} vs. average character error rate: {avg_cer}\n')


    # output model
    # greedyDecoder()
    return 

def train(hparams, trainloader, testloader):
    model = SpeechRecognitionModel(hparams['lstm_input_dim'], hidden_size=hparams['hidden_size'], 
                                   n_cnn_layers=hparams['n_cnn_layers'], n_rnn_layers=hparams['n_rnn_layers'],
                                   n_class=hparams['n_class'])
    optimizer = torch.optim.AdamW(model.parameters(), hparams['learning_rate']) #to ensure that adam has a uniform like effect even with regularisation. 
    #i/e decouples the regularisation term from the gradient updates only subtracting at the end. 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                                    steps_per_epoch= len(trainloader), epochs = hparams['epochs'],
                                                    anneal_strategy='linear')
    criterion = nn.CTCLoss(blank=28) #ie we have 28 characters in our transcription set A-Z, ' ' and . therefore indexed 0->27. We want another one.

    for epoch in hparams['epochs']: 
        epoch_loss = 0
        for batch_idx, batch in enumerate(trainloader):
            print('btach', batch)
            spectograms, labels, input_lengths, label_lengths = batch
            y_hat = model(spectograms)
            output = F.softmax(y_hat, dim=2) #this is batch, time, n_classes output 
            output = output.transpose(0,1) #Wants it in T,N,C
            loss = criterion(output, labels, input_lengths, label_lengths ) #target exp N,S (where S = max(timestep))
            #we need to know the sequence length of each input in the batch for masking (i.e for both label and spetogram).
            # The mask tells us which parts of the sequence were padded and which aren't i.e what our loss function should ignore. 
            #and therefore the ones we actually want to learn on. Padding on both sides for bidrectional context i.e future and past. 
            #padding important for parralelisation whre I think architecture needs to be the same? i.e same # of lstm cells? 
    #also, this is based on the softmax output from the LSTM layer, and therefore, we have the label length as this is what we need to predict, i.e even if blank / same 
    #but the label length is standard. Hence why input is // 2 because of prior convolution in the beginning, not the N residual conv nets.. 
            epoch_loss = epoch_loss + loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if batch_idx % 100 == 0 or batch_idx == len(trainloader.dataset):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectograms), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
    
        with torch.no_grad():
            agg_loss = loss / len(spectograms)
            print(f"TRAIN loss for epoch {epoch}, batch: {batch_idx} : {agg_loss} \n")
            test_incremental(model, testloader, epoch)


    with torch.no_grad():
         torch.save(model.state_dict(), 'weights/weights.pth')

            



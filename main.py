from train import * 
from transcription.transcription import * 
import torchaudio
from torch.utils.data import DataLoader

def main():
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5, 
        "hidden_size": 512, 
        "n_class": 29, #including epsilon (NOTE we allow the model to predict epsilon. This is necessary for CTC)
        #we pass in prev LSTM output so very possible to adjust! 
        "n_feats": 128, 
        "stride": 2,
        "dropout": 0.1, 
        "learning_rate": 5e-4,
        "batch_size": 20, 
        "epochs": 10
    } 

    train_dataset = torchaudio.datasets.LIBRISPEECH('./', url='train-clean-100', download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH('./', url="test-clean", download=True)
    trainloader = DataLoader(dataset=train_dataset, batch_size=hparams["batch_size"], shuffle=True,
                             collate_fn=lambda x: data_preprocessing(x, data_type="train"))
    testloader = DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_preprocessing(x, 'valid'))
    train(hparams=hparams, trainloader=trainloader, testloader=testloader)

    
    






if (__name__ == 'main'):
    main() 
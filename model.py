import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batchNorm1d = nn.BatchNorm1d(embed_size)  # added batch normalization

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batchNorm1d(features)   # added batch normalization
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.wordEmbedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True,
                            dropout = 0)
        # this is the final, output fc layer
        self.linear = nn.Linear(hidden_size, vocab_size)        
    
    def forward(self, features, captions):
        # remove the last token from captions and embed captions
        embeddings = self.wordEmbedding(captions[:,:-1])
        # concatenation and shaping
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstmOut, hidden = self.lstm(embeddings)
        out = self.linear(lstmOut)   
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        samples = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            out = self.linear(out.squeeze(1))
            _, pred = out.max(1)
            samples.append(pred.item())
            inputs = self.wordEmbedding(pred)
            inputs = inputs.unsqueeze(1)
            
        return samples
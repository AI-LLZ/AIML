from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.1):
        super(MLPBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
            nn.Dropout(p),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = -1, n_layers = 1, act_fn = 'LeakyReLU', **kwargs):
        super(Classifier, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        if hidden_dim > 0 and n_layers > 1:
            self.mlp = nn.Sequential(
                MLPBlock(in_dim, hidden_dim),
                *[MLPBlock(hidden_dim, hidden_dim) for _ in range(n_layers - 2)],
                nn.Linear(hidden_dim, out_dim)
            )
        self.act_fn = None
        if act_fn != "None":
            self.act_fn = getattr(nn, act_fn)(**kwargs)
    
    def forward(self, x):
        x = self.mlp(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, p=0.3):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
            nn.Dropout(p),
        )
    def forward(self, inputs):
        return self.block(inputs)

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=2, stride=4, n_channel=128, hidden_dim = -1, n_layers = 1, act_fn = 'LeakyReLU', **kwargs):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(n_input, n_channel, 80, stride=stride),
            nn.MaxPool1d(4),
            ConvBlock(n_channel, n_channel, 3),
            nn.MaxPool1d(4),
            ConvBlock(n_channel, 2 * n_channel, 3),
            nn.MaxPool1d(4),
            ConvBlock(2 * n_channel, 4 * n_channel, 3),
            nn.MaxPool1d(4),
        )
        
        self.cls = Classifier(
            4 * n_channel, n_output, hidden_dim = hidden_dim, n_layers = n_layers, act_fn = act_fn, **kwargs
        )

    def forward(self, x):
        print(x.shape, 'at input')
        x = self.feature_extractor(x)
        print(x.shape)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze(-1)
        x = self.cls(x)
        return x
class M18(nn.Module):
    def __init__(self, n_input=1, n_output=2, stride=4, n_channel=64, hidden_dim = -1, n_layers = 1, act_fn = 'LeakyReLU', **kwargs):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(n_input, n_channel, 80, stride=stride),
            nn.MaxPool1d(4),
            *[ConvBlock(n_channel, n_channel, 3) for _ in range(4)],
            nn.MaxPool1d(4),
            ConvBlock(n_channel, 2 * n_channel, 3),
            *[ConvBlock(2 * n_channel, 2 * n_channel, 3) for _ in range(3)],
            nn.MaxPool1d(4),
            ConvBlock(2 * n_channel, 4 * n_channel, 3),
            *[ConvBlock(4 * n_channel, 4 * n_channel, 3) for _ in range(3)],
            nn.MaxPool1d(4),
            ConvBlock(4 * n_channel, 8 * n_channel, 3),
            *[ConvBlock(8 * n_channel, 8 * n_channel, 3) for _ in range(3)],
            nn.MaxPool1d(4),
        )
        
        self.cls = Classifier(
            8 * n_channel, n_output, hidden_dim = hidden_dim, n_layers = n_layers, act_fn = act_fn, **kwargs
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        print(x.shape)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze(-1)
        x = self.cls(x)
        return x

class Wav2Vec2(nn.Module):
    def __init__(self, config = None, pretrained = True, load_model = 'superb/wav2vec2-base-superb-ks'):
        super().__init__()
        if config is None and not pretrained:
            raise RuntimeError("Config is required if pretrain weight is not loaded.")
       
        if config is not None:
            self.model = Wav2Vec2ForSequenceClassification(config)
        elif pretrained:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(load_model)
        self.model.gradient_checkpointing_enable()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    x = torch.randn((72, 1, 96000))
    model = M5(1, 7, 4, 128)
    print(model)
    o = model(x)
    print(o.shape)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    n = count_parameters(model)
    print("Number of parameters of M5: %s" % round(n / 1e6, 1))
    
    print(x.shape)
    model = M18(1,2)
    
    o = model(x)
    print(o.shape)
    n = count_parameters(model)
    print("Number of parameters of M18: %s" % round(n / 1e6, 1))


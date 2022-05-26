import torch
from torch import nn
import torch.nn.functional as F

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.5):
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


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32, hidden_dim = -1, n_layers = 1, act_fn = 'LeakyReLU', **kwargs):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_input, n_channel, 80, stride=stride),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, n_channel, 3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, 2 * n_channel, 3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * n_channel, 2 * n_channel, 3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        
        self.cls = Classifier(
            2 * n_channel, n_output, hidden_dim = 8, n_layers = 3, act_fn = 'Sigmoid', **kwargs
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze(-1)
        x = self.cls(x)
        return x

class Wav2Vec2(nn.Module):
    def __init__(self, config = None, pretrained = True, load_model = 'Wav2Vec2PreTrainedModel'):
        super().__init__()
        if config is None and not pretrained:
            raise RuntimeError("Config is required if pretrain weight is not loaded.")
       
        if pretrained:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks")
        if config is not None:
            self.model = Wav2Vec2ForSequenceClassification(config)
        self.model.gradient_checkpointing_enable()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    x = torch.randn((72, 1, 96000))
    model = M5(1, 7)
    print(model)
    o = model(x)
    print(o.shape)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    n = count_parameters(model)
    print("Number of parameters of M5: %s" % n)
    from transformers import Wav2Vec2Config
    
    x = x.squeeze(1)
    print(x.shape)
    config = Wav2Vec2Config(num_hidden_layers=6, num_attention_heads=6, num_labels=2)
    model = Wav2Vec2(config)
    
    o = model(x)
    print(o.shape)
    n = count_parameters(model)
    print("Number of parameters of Wav2Vec2: %s" % n)


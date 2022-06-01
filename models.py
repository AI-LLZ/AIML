from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


def init_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_normal_(module.weight.data)


class vggish_params:
    # Architectural constants.
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.

    # Hyperparameters used in feature and example generation.
    SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

    # Parameters used for embedding postprocessing.
    PCA_EIGEN_VECTORS_NAME = "pca_eigen_vectors"
    PCA_MEANS_NAME = "pca_means"
    QUANTIZE_MIN_VAL = -2.0
    QUANTIZE_MAX_VAL = +2.0

    # Hyperparameters used in training.
    INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
    LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
    ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

    # Names of ops, tensors, and features.
    INPUT_OP_NAME = "vggish/input_features"
    INPUT_TENSOR_NAME = INPUT_OP_NAME + ":0"
    OUTPUT_OP_NAME = "vggish/embedding"
    OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ":0"
    AUDIO_EMBEDDING_FEATURE_NAME = "audio_embedding"


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, p=0):
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
    def __init__(
        self, in_dim, out_dim, hidden_dim=-1, n_layers=1, act_fn="LeakyReLU", **kwargs
    ):
        super(Classifier, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        if hidden_dim > 0 and n_layers > 1:
            self.mlp = nn.Sequential(
                MLPBlock(in_dim, hidden_dim),
                *[MLPBlock(hidden_dim, hidden_dim) for _ in range(n_layers - 2)],
                nn.Linear(hidden_dim, out_dim),
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
            nn.ReLU(),
            nn.Dropout(p),
        )

    def forward(self, inputs):
        return self.block(inputs)


class M5(nn.Module):
    def __init__(
        self,
        n_input=1,
        n_output=2,
        stride=4,
        n_channel=128,
        hidden_dim=-1,
        n_layers=1,
        act_fn="LeakyReLU",
        **kwargs
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(n_input, n_channel, 80, stride=stride, p=0),
            nn.MaxPool1d(4),
            ConvBlock(n_channel, n_channel, 3, p=0),
            nn.MaxPool1d(4),
            ConvBlock(n_channel, 2 * n_channel, 3, p=0),
            nn.MaxPool1d(4),
            ConvBlock(2 * n_channel, 4 * n_channel, 3, p=0),
            nn.MaxPool1d(4),
        )

        self.cls = Classifier(
            4 * n_channel,
            n_output,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            act_fn=act_fn,
            **kwargs,
        )
        self.apply(init_weight)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze(-1)
        x = self.cls(x)
        return x

    def get_embedding(self, x):
        x = self.feature_extractor(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze(-1)
        return x


class M18(nn.Module):
    def __init__(
        self,
        n_input=1,
        n_output=2,
        stride=4,
        n_channel=64,
        hidden_dim=-1,
        n_layers=1,
        act_fn="LeakyReLU",
        **kwargs
    ):
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
            8 * n_channel,
            n_output,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            act_fn=act_fn,
            **kwargs,
        )
        self.apply(init_weight)

    def forward(self, x):
        x = self.feature_extractor(x)
        # print(x.shape)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze(-1)
        x = self.cls(x)
        return x


class Wav2Vec2(nn.Module):
    def __init__(
        self, config=None, pretrained=True, load_model="superb/wav2vec2-base-superb-ks"
    ):
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


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.
    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (
                vggish_params.EMBEDDING_SIZE,
                vggish_params.EMBEDDING_SIZE,
            ),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(
            self.pca_eigen_vectors, requires_grad=False
        )
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.
        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.
        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(
            self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)
        ).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.
    Adapted from: https://github.com/harritaylor/torch-vggish
    The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
    improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
    correction for flattening in the fully-connected layers.
    """

    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # x = torch.randn((72, 1, 96000))
    # model = M5(1, 7, 4, 128)
    # print(model)
    # o = model(x)
    # print(o.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



    model = VGGish(pretrained=False)
    print(model)
    i = torch.randn([5*72, 1, 96, 64])
    o = model(i)
    print(o.shape)

    n = count_parameters(model)
    print("Number of parameters of VGGish: %s" % round(n / 1e6, 1))
    # model = M18(1,2)

    # o = model(x)
    # print(o.shape)
    # n = count_parameters(model)
    # print("Number of parameters of M18: %s" % round(n / 1e6, 1))

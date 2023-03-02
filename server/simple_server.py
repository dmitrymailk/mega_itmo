# import main Flask class and request object
from flask import Flask, request, send_file
from flask_cors import CORS

from PIL import Image
from io import BytesIO
import base64

import pytorch_lightning as pl

# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms

from omegaconf import OmegaConf
from IPython.display import Audio, display
import os

torch.hub.download_url_to_file(
    "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml",
    "latest_silero_models.yml",
    progress=False,
)
models = OmegaConf.load("latest_silero_models.yml")

device = torch.device("cpu")
torch.set_num_threads(4)
local_file = "model.pt"

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file(
        "https://models.silero.ai/models/tts/ru/v3_1_ru.pt", local_file
    )

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = "В недрах тундры выдры в г+етрах т+ырят в вёдра ядра кедров."
sample_rate = 48000
speaker = "baya"


class VGGNet(nn.Module):
    def __init__(self, n_classes=None, conv_arch=None):
        super(VGGNet, self).__init__()
        self.conv_arch = conv_arch
        self.n_classes = n_classes

        self.conv = self.get_conv()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # The fully-connected part
            nn.Linear(conv_arch[-1][1] * 4 * 4, 200),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(100),
            nn.Linear(100, n_classes),
        )

    def get_conv(self):
        conv_blks = []
        in_channels = 3
        # The convolutional part
        for num_convs, out_channels in self.conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(*conv_blks)

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.Dropout(0.5),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


class LitModel2(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        num_classes,
        learning_rate=2e-4,
    ):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        conv_arch = ((2, 64), (2, 128), (3, 256))
        self.model = VGGNet(num_classes, conv_arch=conv_arch)

        self.accuracy = Accuracy("multiclass", num_classes=num_classes)

    # will be used during inference
    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


modelEmotion = LitModel2.load_from_checkpoint(
    "/home/dimweb/Desktop/sandbox/mega_itmo/face_recognition_experiments/wandb-lightning/zrtotnhv/checkpoints/epoch=49-step=2550.ckpt"
)
modelEmotion.eval()


def predict_emotion(model, image):
    emotions = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    }

    transform_image = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
        ]
    )

    image = transform_image(image)
    image = image.unsqueeze(0)

    # predict with the model
    print(image.shape)
    y_hat = model(image)
    pred = int(y_hat.argmax())
    print(emotions[pred])
    return emotions[pred]


# create the Flask app
app = Flask(__name__)
CORS(app)


def create_dumb_phrase(emotion_state):
    if emotion_state == "Neutral":
        return "Согласно моему анализу вероятно у вас все хорошо, так держать!"
    elif emotion_state == "Fear":
        return "Вам страшно??"
    elif emotion_state == "Angry":
        return "Гнев это не очень здорово, постарайтесь успокоится"
    elif emotion_state == "Sad":
        return "Не грусти, че такой грустный?"
    else:
        return "У вас всё хорошо?"


@app.route("/analyse-mood/", methods=["POST"])
def query_example():
    if request.method == "POST":
        content = request.json
        image = str(content["image"])
        image = image.replace("data:image/jpeg;base64,", "")
        image = Image.open(BytesIO(base64.b64decode(image)))
        image.save("test.jpg")
        emotion = predict_emotion(
            model=modelEmotion,
            image=image,
        )
        phrase = create_dumb_phrase(emotion_state=emotion)
        audio_paths = model.save_wav(
            text=phrase,
            speaker=speaker,
            sample_rate=sample_rate,
        )

        wav_base64 = base64.b64encode(open("./test.wav", "rb").read())

        return wav_base64

    return f"Был получен {request.method} запрос."


if __name__ == "__main__":
    app.run(debug=True, port=5000)

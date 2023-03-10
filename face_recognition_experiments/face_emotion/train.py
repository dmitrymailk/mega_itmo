import pytorch_lightning as pl

# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import FER2013
from torchvision.models import SqueezeNet


import wandb

wandb.login()


class FER2013DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir: str = "/home/dimweb/Desktop/sandbox/mega_itmo/face_recognition_experiments",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform_train = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
            ]
        )

        self.num_classes = 7

    def prepare_data(self):
        FER2013(
            self.data_dir,
            split="train",
        )
        FER2013(
            self.data_dir,
            split="test",
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            data_train = FER2013(
                self.data_dir,
                split="train",
                transform=self.transform_train,
            )
            train_size = int(len(data_train) * 0.9)
            val_size = len(data_train) - train_size
            split_indices = [train_size, val_size]
            self.data_train, self.data_val = random_split(data_train, split_indices)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=10,
        )


class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(
                        val_imgs[: self.num_samples],
                        preds[: self.num_samples],
                        val_labels[: self.num_samples],
                    )
                ]
            }
        )


class LitModel(pl.LightningModule):
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

        self.squeeze_net = SqueezeNet(
            num_classes=num_classes,
        )

        self.accuracy = Accuracy("multiclass", num_classes=num_classes)

    # will be used during inference
    def forward(self, x):
        x = self.squeeze_net(x)
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


if __name__ == "__main__":
    dm = FER2013DataModule(
        batch_size=512,
    )
    dm.prepare_data()
    dm.setup()
    model = LitModel2(
        input_shape=(3, 32, 32),
        num_classes=dm.num_classes,
    )

    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]

    wandb_logger = WandbLogger(project="wandb-lightning", job_type="train")

    # Initialize Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=80,
        gpus=1,
        logger=wandb_logger,
        callbacks=[
            # early_stop_callback,
            ImagePredictionLogger(val_samples),
            checkpoint_callback,
        ],
    )

    trainer.fit(model, dm)

    # Close wandb run
    wandb.finish()

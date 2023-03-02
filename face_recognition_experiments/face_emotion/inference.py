from face_emotion.train import LitModel2

import torch
import torchvision
from torchvision import transforms
from PIL import Image

# from keras.models import load_model


if __name__ == "__main__":
    frame = "torch"
    if frame == "torch":
        model = LitModel2.load_from_checkpoint(
            "/home/dimweb/Desktop/sandbox/mega_itmo/face_recognition_experiments/wandb-lightning/zrtotnhv/checkpoints/epoch=49-step=2550.ckpt"
        )
        model.eval()

        emotions = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral",
        }

        image = Image.open(
            "/home/dimweb/Desktop/sandbox/mega_itmo/face_recognition_experiments/img/image_6.png",
        )
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
    else:
        model = load_model(
            "/home/dimweb/Desktop/sandbox/mega_itmo/face_recognition_experiments/top_model/epoch_75.hdf5"
        )
        emotions = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral",
        }

        image = Image.open(
            "/home/dimweb/Desktop/sandbox/mega_itmo/face_recognition_experiments/img/image_5.png",
        )
        transform_image = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        image = transform_image(image)

        # predict with the model
        print(image.shape)
        y_hat = model.predict(image.numpy())
        pred = int(y_hat.argmax())
        print(emotions[pred])

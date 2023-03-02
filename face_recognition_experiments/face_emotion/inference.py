from face_emotion.train import LitModel

import torch
import torchvision
from torchvision import transforms
from PIL import Image

if __name__ == "__main__":
    model = LitModel.load_from_checkpoint(
        "/home/dimweb/Desktop/sandbox/mega_itmo/face_recognition_experiments/wandb-lightning/4jxffekz/checkpoints/epoch=20-step=2121.ckpt"
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

    model.eval()
    image = Image.open(
        "/home/dimweb/Desktop/sandbox/mega_itmo/face_recognition_experiments/img/image_3.png",
    )
    transform_image = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    image = transform_image(image)
    image = image.unsqueeze(0)

    # predict with the model
    print(image.shape)
    y_hat = model(image)
    pred = int(y_hat.argmax())
    print(emotions[pred])

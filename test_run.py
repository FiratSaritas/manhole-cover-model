import unittest
import run
import torch
import argparse
import PIL
from PIL import Image
from torchvision import transforms as transforms
import json
from utils.prediction.pred_interface import PredictionInterface
label_dict = {
    'Rost/Strassenrost': 0,
    'Vollguss/Pickelloch belueftet': 1,
    'Gussbeton/Pickelloch geschlossen': 2,
    'Vollguss/Pickelloch geschlossen': 3,
    'Gussbeton/Pickelloch belueftet': 4,
    'Vollguss/Handgriff geschlossen': 5,
    'Gussbeton/Handgriff seitlich': 6,
    'Rost/Einlauf rund': 7,
    'Rost/Strassenrost gewoelbt': 8,
    'Vollguss/Aufklappbar': 9,
    'Gussbeton/Handgriff mitte': 10,
    'Vollguss/Handgriff geschlossen, verschraubt': 11
}
CROP_FACTOR = 1.3
model_ = torch.load('model/model.pth',map_location ='cpu')
model = PredictionInterface(model_)
my_test_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )

class TestRun(unittest.TestCase):
    def test_predict_one(self):
        image = run.image_to_tensor("test_image.jpg")
        result = model.predict_one(image,label_dict,top_n=1)
        pred_type = type(result)
        self.assertEqual(pred_type, dict)
import torch
import argparse
from utils import MHCoverDataset, get_dataloader
import PIL
from PIL import Image
from torchvision import transforms as transforms
import json



label_dict = {'Rost/Strassenrost': 0,
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
              'Vollguss/Handgriff geschlossen, verschraubt': 11}

CROP_FACTOR = 1.3
model = torch.load('model/model_resnet152.pth',map_location ='cpu')

my_test_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

parser = argparse.ArgumentParser(description="""
    This script is going to predict the class of an image.
    """)
parser.add_argument("url", help="URL of the image")
parser.add_argument("top_n", help="Number of predictions", nargs='?', type=int, const=1, default=1)



def crop_center(pil_img, crop_width, crop_height):
    """
    Crop the center of the image.

    Arguments:
    pil_img -- PIL image
    crop_width -- width of the crop
    crop_height -- height of the crop
    Returns:
    PIL image
    """
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def image_to_tensor(URL: str):
    """
    Converts image to tensor

    Args:
        URL (str): _description_

    Returns:
        _type_: _description_
    """
    image = PIL.Image.open(URL)
    height, width = image.size
    image = crop_center(image, width / CROP_FACTOR, height / CROP_FACTOR)
    image = image.resize(size=(400, 400),resample=Image.Resampling.NEAREST)
    image = my_test_transforms(image)
    image = image.unsqueeze(0)  
    return image


def main(args):
    try:
        image = image_to_tensor(args.url)
    except:
        print("error, could not load image, please try again")
        exit()

    try :
        TOP_N = int(args.top_n)
    except ValueError:
        print("please enter a number")
        exit()
    if TOP_N > 12:
        print("please enter a number less than 13")
        exit()
    if TOP_N < 1:
        print("please enter a number greater than 0")
        exit()
    result = model.predict_one(image,label_dict,top_n=TOP_N)
    with open('result.json', 'w') as fp:
        json.dump(result, fp)
    print(result)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args=args)
import torch
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
model = torch.load('model/model_resnet152.pth')


my_test_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

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
    image = image.resize(size=(400, 400),resample=Image.NEAREST)
    image = my_test_transforms(image)
    image = image.unsqueeze(0)  
    return image



if __name__ == '__main__':
    #Eingabe-Informationen
    url= input("get url of image: ")
    try:
        image = image_to_tensor(url)
    except:
        print("error, could not load image, please try again")
        exit()
    top_n = input("get number of top predictions: ")
    try :
        top_n = int(top_n)
    except ValueError:
        print("please enter a number")
        exit()
    if top_n > 12:
        print("please enter a number less than 13")
        exit()
    result = model.predict_one(image,label_dict,top_n=top_n)
    with open('result.json', 'w') as fp:
        json.dump(result, fp)
    print(result)
<a href=""><img src="https://img.shields.io/badge/status-online-green" /></a>

# Manhole Cover Classifier

![ezgif-4-b657bbd0b1](https://user-images.githubusercontent.com/82641568/169886621-3f91f6a3-67a1-49ad-af67-cbfbcc163f6d.gif)
![ezgif-5-35c28f41eb](https://user-images.githubusercontent.com/82641568/169886608-2c9da711-92db-4796-a9d5-5bdedecb9ab8.gif)
![ezgif-4-1abd2a1947](https://user-images.githubusercontent.com/82641568/169886629-a17029e4-c58d-4ac6-af37-b0026cf04415.gif)


This repository predicts the class of a Manhole Cover image from 12 different classes:
  - Rost/Strassenrost
  - Vollguss/Pickelloch belueftet
  - Gussbeton/Pickelloch geschlossen
  - Vollguss/Pickelloch geschlossen
  - Gussbeton/Pickelloch belueftet
  - Vollguss/Handgriff geschlossen
  - Gussbeton/Handgriff seitlich
  - Rost/Einlauf rund
  - Rost/Strassenrost gewoelbt
  - Vollguss/Aufklappbar
  - Gussbeton/Handgriff mitte
  - Vollguss/Handgriff geschlossen, verschraubt
  

## Installation

1. Clone project locally 

```bash
git clone git@github.com:FiratSaritas/manhole-cover-model.git
```

2. Download model from Google Drive and add it to the folder `./model` here: https://drive.google.com/file/d/1JYONs6lFC2mbSC2KOwpnGopcbL1OTXBc/view?usp=sharing 


3. Install required packages

Preferably you create a new enviorment (conda environment is also possible).

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py [Url of image]
```

Output:
  - result.json-file in the current folder
  
Additional Usage with parameter:

```bash
python run.py --help
```

## Examples

```bash
python run.py image.png 
```
or with number of top predictions:

```bash
python run.py image.png 3
```

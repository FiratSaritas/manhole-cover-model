<a href=""><img src="https://img.shields.io/badge/status-online-green" /></a>

# Manhole Cover Classifier

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
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

## Usage

```bash
python run.py [Url of image] [Number of top predictions]
```

Output:
  - result.json-file in the current folder
  

## Examples

```bash
python run.py image.png 3
```
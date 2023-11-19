# tichu-card-detection
Generating a dataset of Tichu playing cards to train a neural net. Taking [geaxgx's work](https://github.com/geaxgx/playing-card-detection) and applying it to the Tichu card playing game.

This dataset can be used for the training of a neural net intended to detect/localize playing cards. It was used on the project __[Playing card detection with YOLO v3](https://youtu.be/pnntrewH0xg)__

## Changes from the original 
The notebook ```creating_playing_cards_dataset.ipynb``` is a guide through the creation of a dataset of playing cards.
 - The script was modified to extract card images from pictures of cards instead of from videos.
 - Instead of having to hardcode the card names, they are extracted from the directory names containing the card images

## Card Generation Workflow
1. Generate the Training and Validation datasets with the ```creating_playing_cards_dataset.ipynb``` notebook.
    1. 
1. Train the yolov5 Model
    ```
    $ cd yolov5
    $ python train.py --img 640 --epochs 3 --data ../data/scenes/dataset.yaml --weights yolov5s.pt
    ```
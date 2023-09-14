# Long-Term Point Tracking with PIPs++

This is the official code release for the models in our ICCV 2023 paper, "PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking".
**[[Paper](https://arxiv.org/abs/2307.15055)] [[Project Page](https://pointodyssey.com/)]**

<img src='https://pointodyssey.com/assets/point_tracks.jpg'>

## Requirements

The lines below should set up a fresh environment with everything you need: 

```
conda create -n py38 python=3.8
conda activate py38
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Demo

To download our reference model, run this line:

```
sh get_reference_model.sh
```

Or, use the dropbox link inside that file. 

...


## Model implementation

To inspect PIPs++, the main file to look at is `nets/pips2.py`.


## PointOdyssey preprocessing

...

## Training

...


## Testing

...



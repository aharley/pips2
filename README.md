# Long-Term Point Tracking with PIPs++

This is the official code release for the PIPs++ model presented in our ICCV 2023 paper, "PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking".

**[[Paper](https://arxiv.org/abs/2307.15055)] [[Project Page](https://pointodyssey.com/)]**

<img src='https://pointodyssey.com/assets/point_tracks.jpg'>

## Requirements

The lines below should set up a fresh environment with everything you need: 

```
conda create -n pips2 python=3.8
conda activate pips2
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


## PointOdyssey 

We train our model on the [PointOdyssey](https://huggingface.co/datasets/aharley/pointodyssey) dataset.

With a standard dataloader (e.g., `datasets/pointodysseydataset.py`), loading PointOdyssey's high-resolution images can be a bottleneck at training time. To speed things up, we export mp4 clips from the dataset, at the resolution we want to train at, with augmentations baked in. To do this, run:

```
python export_mp4_dataset.py
```

This will produce a dataset of clips, in `pod_export/$VERSION_$SEQLEN`. The script will also produce some temporary folders to help write the data; these can be safely deleted afterwards. The script should also be safe to run in multiple threads in parallel. Depending on disk speeds, writing the full dataset with 4 threads should take about 24h. As soon as you have a few clips, however, you can start training.


## Training

...


## Testing

...



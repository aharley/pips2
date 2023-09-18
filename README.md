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

To try this model on a sample video, run this:

```
python demo.py
```

This will run the model on the video included in `stock_videos/`.

For each 8-frame subsequence, the model will return `trajs_e`. This is estimated trajectory data for a set of points, shaped `B,S,N,2`, where `S` is the sequence length and `N` is the number of particles, and 2 is the x and y coordinates. The script will also produce tensorboard logs with visualizations, which go into `logs_demo/`.

In the tensorboard for `logs_demo/` you should be able to find visualizations like this:

<img src='https://pointodyssey.com/camel_compressed.gif'>


## PointOdyssey 

We train our model on the [PointOdyssey](https://huggingface.co/datasets/aharley/pointodyssey) dataset.

With a standard dataloader (e.g., `datasets/pointodysseydataset.py`), loading PointOdyssey's high-resolution images can be a bottleneck at training time. To speed things up, we export mp4 clips from the dataset, at the resolution we want to train at, with augmentations baked in. To do this, run:

```
python export_mp4_dataset.py
```

This will produce a dataset of clips, in `pod_export/$VERSION_$SEQLEN`. The script will also produce some temporary folders to help write the data; these can be safely deleted afterwards. The script should also be safe to run in multiple threads in parallel. Depending on disk speeds, writing the full dataset with 4 threads should take about 24h.

The script output should look something like this: 
```
.36_A_em00_aa_110247; step 000676/153264; this_step 018524; itime 3.77
.36_A_em00_aa_110247; step 000677/153264; this_step 017116; itime 2.74
.36_A_em00_aa_110247; step 000678/153264; this_step 095616; itime 6.11
sum(~mot_ok) 276
xN=0
sum(~mot_ok) 2000
N=0
:::sum(~mot_ok) 14
.36_A_em00_aa_110247; step 000685/153264; this_step 002960; itime 6.51
.36_A_em00_aa_110247; step 000686/153264; this_step 034423; itime 6.91
```

Note that the clips are produced in random order. The script is fairly friendly to multiple parallel runs, and avoids re-writing mp4s that have already been produced. Sometimes sampling from PointOdyssey will fail, and the script will report the reason for the failure (e.g., no valid tracks after applying augmentations).

As soon as you have a few exported clips, you can start playing with the trainer. The trainer will load the exported data using `dataset/exportdataset.py`.


## Training

To train a model, simply run `train.py`.

It should first print some diagnostic information about the model and dataset. Then it should print a message for each training step, indicating the model name, progress, read time, iteration time, and loss. 

```
model_name 4_36_128_i6_5e-4s_A_aa03_113745
loading export...
found 57867 folders in pod_export/aa_36
+--------------------------------------------------------+------------+
|                        Modules                         | Parameters |
+--------------------------------------------------------+------------+
|           module.fnet.layer3.0.conv1.weight            |   110592   |
|           module.fnet.layer3.0.conv2.weight            |   147456   |
|           module.fnet.layer3.1.conv1.weight            |   147456   |
|           module.fnet.layer3.1.conv2.weight            |   147456   |
|           module.fnet.layer4.0.conv1.weight            |   147456   |
|           module.fnet.layer4.0.conv2.weight            |   147456   |
|           module.fnet.layer4.1.conv1.weight            |   147456   |
|           module.fnet.layer4.1.conv2.weight            |   147456   |
|                module.fnet.conv2.weight                |   958464   |
|    module.delta_block.first_block_conv.conv.weight     |   275712   |
| module.delta_block.basicblock_list.2.conv2.conv.weight |   196608   |
| module.delta_block.basicblock_list.3.conv1.conv.weight |   196608   |
| module.delta_block.basicblock_list.3.conv2.conv.weight |   196608   |
| module.delta_block.basicblock_list.4.conv1.conv.weight |   393216   |
| module.delta_block.basicblock_list.4.conv2.conv.weight |   786432   |
| module.delta_block.basicblock_list.5.conv1.conv.weight |   786432   |
| module.delta_block.basicblock_list.5.conv2.conv.weight |   786432   |
| module.delta_block.basicblock_list.6.conv1.conv.weight |  1572864   |
| module.delta_block.basicblock_list.6.conv2.conv.weight |  3145728   |
| module.delta_block.basicblock_list.7.conv1.conv.weight |  3145728   |
| module.delta_block.basicblock_list.7.conv2.conv.weight | 3145728  |
+--------------------------------------------------------+------------+
total params: 17.57 M
4_36_128_i6_5e-4s_A_aa03_113745; step 000001/200000; rtime 3.69; itime 5.63; loss 35.030; loss_t 35.03; d_t 1.8; d_v nan
4_36_128_i6_5e-4s_A_aa03_113745; step 000002/200000; rtime 0.00; itime 1.45; loss 31.024; loss_t 33.03; d_t 2.5; d_v nan
4_36_128_i6_5e-4s_A_aa03_113745; step 000003/200000; rtime 0.00; itime 1.45; loss 30.908; loss_t 32.32; d_t 2.7; d_v nan
4_36_128_i6_5e-4s_A_aa03_113745; step 000004/200000; rtime 0.00; itime 1.45; loss 31.327; loss_t 32.07; d_t 2.8; d_v nan
4_36_128_i6_5e-4s_A_aa03_113745; step 000005/200000; rtime 0.00; itime 1.45; loss 29.762; loss_t 31.61; d_t 2.9; d_v nan
[...]
 ```

The final items in each line, `d_t` and d_v`, show the result of the `d_avg` metric on the training set and the validation set. Note that `d_v` will show `nan` until the first validation step.

To reproduce the reference model, you should train for 200k iterations (using the fully-exported dataset), with `B=4, S=36, crop_size=(256,384)`. Then, fine-tune for 10k iterations using higher resolution and longer clips: `B=1, S=64, crop_size=(512,896)`. If you can afford a higher batch size, you should use it. For this high-resolution finetuning, you can either export new mp4s, or use `pointodysseydataset.py` directly. 


## Testing

We provide evaluation scripts for all of the datasets reported in the paper. The values in this repo are slightly different than those in the PDF, largely because we fixed some bugs in the dataset and re-trained the model for this release.

**TAP-VID-DAVIS**

For each point with a valid annotation in frame0, we track it to the end of the video (<200 frames). The data comes from the [DeepMind TAP-NET repo](https://github.com/google-deepmind/tapnet#tap-vid-benchmark). 

With the reference model, `test_on_tap.py` should yield `d_avg 70.6; survival_16 89.3; median_l2 6.9`.

**CroHD**

We chop the videos in to 1000-frame clips, and track heads from the beginning to the end. The data comes from the "Get all data" link on the [Head Tracking 21 MOT Challenge](https://motchallenge.net/data/Head_Tracking_21/) page. Downloading and unzipping that should give you the folders HT21 and HT21Labels, which our dataloader relies on.

With the reference model, `test_on_cro.py` should yield `d_avg 50.6.5; survival_16 50.6; median_l2 19.7`.

**PointOdyssey test set**

For each point with a valid annotation in frame0, we track it to the end of the video (~2k frames). Note that here we use the `pointodysseydataset_fullseq.py` dataloader, and we load `S=128` frames at a time, because 2k frames will not fit in memory. 

With the reference model, `test_on_pod.py` should yield `d_avg 31.3; survival_16 32.7; median_l2 33.0`. 


## Citation

If you use this code for your research, please cite:

**PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking.** Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein, Leonidas J. Guibas. In ICCV 2023.

Bibtex:
```
@inproceedings{zheng2023point,
 author = {Yang Zheng and Adam W. Harley and Bokui Shen and Gordon Wetzstein and Leonidas J. Guibas},
 title = {PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking},
 booktitle = {ICCV},
 year = {2023}
}
```


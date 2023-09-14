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

This will produce a dataset of clips, in `pod_export/$VERSION_$SEQLEN`. The script will also produce some temporary folders to help write the data; these can be safely deleted afterwards. The script should also be safe to run in multiple threads in parallel. Depending on disk speeds, writing the full dataset with 4 threads should take about 24h.


As soon as you have a few clips, you can start training. The trainer will load the data using `dataset/exportdataset.py`.


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

The final items, `d_t` and d_v` show the result of the `d_avg` metric on the training set and the validation set. Note that `d_v` will show `nan` until the first validation step.

To reproduce the reference model, you should train for about 200k iterations with `B=4, S=36, crop_size=(256,384)`. Then, fine-tune for about 10k iterations with `B=1, S=64, crop_size=(512,896)`. If you can afford a higher batch size, you should use it. For this high-resolution finetuning, you can either export new mp4s, or use `pointodysseydataset.py` directly. 


## Testing

...



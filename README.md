

## Model

To download our reference model, run this line:

```
sh get_reference_model.sh
```

Or, use the dropbox link inside that file. 


## Requirements

The lines below should set up a fresh environment with everything you need: 

```
conda create -n py38 python=3.8
conda activate py38
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
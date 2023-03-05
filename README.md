This is a stand-alone self-contained Python project to train, run, and evaluate an experimental model proposed by myself to check if we can get rid of the U-Net structure and purely use the Transformer architecture to do semantic segmentation on medical images. The result is quite promising as it achieves an even-slightly-better performance with nearly only half of the parameters comparing to the state-of-the-art (at the time of writing) [SwinUNETR](https://arxiv.org/abs/2201.01266). Please refer to the [summary essey](/MedViT.pdf) for details.

# Prerequisite

- Install [conda](https://docs.conda.io/en/latest/).

# Prepare

First prepare a Python virtual environment by

```
conda create --name MedViT python=3.11
conda activate MedViT
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
```

Test your environment by running

```
python -c "import torch; print(torch.cuda.device_count())"
```

which should show the correct number of GPUs.

Then create a local env file by

```
cp .env .env.local
```

and set `MEDVIT_WORKSPACE` to be the path of some directory with sufficiently large space, and `MEDVIT_DATA_ROOT` to be the uncompressed folder of the [BraTS2021](http://braintumorsegmentation.org/) dataset containing subfolders like

```
- BraTS2021_00001/
- BraTS2021_00002/
- BraTS2021_00003/
...
```

# Pipeline

The codebase provides a reasonable default setting. Run following commands in turn to train, predict, and evaluate.

```
make train
make predict
make evaluate
```

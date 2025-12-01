# Chase: Self-Improvement Policy by Challenging Case Selection

## Installation

1. Create a new conda environment
```bash
conda create -n sime python=3.9
conda activate sime
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Install [Pytorch3D](https://github.com/facebookresearch/pytorch3d) from source.
```bash
mkdir dependencies && cd dependencies
git clone git@github.com:facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ../..
```

4. Install robomimic
```bash
cd dependencies
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
cd ../..
```

## Reproducing Benchmark Results

### Datasets

We use the [robomimic](https://github.com/ARISE-Initiative/robomimic) benchmark for simulation experiments. Please follow the [instructions](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html) to download the low-dim datasets.
```bash
cd simulation && python download_datasets.py --tasks sim --dataset_types ph --hdf5_types low_dim --download_dir datasets
```



```bash
cd simulation
# lift
python run_full.py --dataset datasets/lift/ph/low_dim_v141.hdf5 --output_dir out/lift/ --used_demo core_20 --task lift  --noise_scale 0.01
# can
python run_full.py --dataset datasets/can/ph/low_dim_v141.hdf5 --output_dir out/can/ --used_demo core_20 --task can  --noise_scale 0.01
# square
python run_full.py --dataset datasets/square/ph/low_dim_v141.hdf5 --output_dir out/square/ --used_demo core_20 --task square 

```

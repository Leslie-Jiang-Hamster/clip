# Installation

## Clone this repo

```bash
git clone https://github.com/Leslie-Jiang-Hamster/clip
cd clip
```

## Installing conda

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Enter a lot of yes, then you'll see "Thank you for installing Miniconda3!"

```bash
source ~/.bashrc
```

Test your installation by running `conda list`. If conda has been installed correctly, a list of installed packages appears.

Then create an environment called clip:

```bash
conda create -n clip python=3.9
conda activate clip
```

## Installing huggingface transformers

(Ensure that you are in the "clip" conda environment) Run:

```bash
conda install pip
pip install 'transformers[torch]'
pip install pillow
```

## Test

```bash
python clip.py
```

This program will predict if the image "cat.jpg" is a cat, a dog or a horse.
If the output is cat, then we are finished.

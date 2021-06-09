# Cosmic Microwave Background Source Separation
This repository contains source codes for the Cosmic Microwave Background (CMB) source separation problem

## Installation
You need Python 3 to run the program. You have to install required libraries beforehand by running:
```
# If you are using pip
pip install -r requirements.txt

# If you are using conda
conda install --file requirements.txt
```

## Usage
Evaluate different methods on different levels with `evaluation.py`:
```
python evaluation.py -l <level> -m <method>

l: Level, integer value from 1 to 10, default value = 1
m: method, either 'std' for standard solution (matrix multiplication) or 'cg' for conjugate gradient method, default value = 'std'
```

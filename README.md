# YeastInference

## Installation
These instructions have only been tested on Ubuntu, but should also work on other operating systems.
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.7
2. Create a new conda environment and activate it:
    ```bash
    conda create -n yeast-env python
    conda activate yeast-env 
    
    ```
3. Install pytorch-cpu:
    
    On Windows & Linux:

    ```bash
    conda install pytorch torchvision cpuonly -c pytorch
    
    ```

    On Mac:
 
    ```bash
    conda install pytorch torchvision -c pytorch
    
    ```

4. Clone this repo and install dependencies:
    ```bash
    git clone https://github.com/imagirom/YeastInference.git
    cd YeastInference
    pip install -r requirements.txt
    pip install -e git+https://github.com/inferno-pytorch/inferno@cache-reject#egg=inferno-0.3.1
    ```

## Usage

Place the TIF files you want to process in a new folder `YeastInference/data`. In the `YeastInference` folder, run
```bash
python predict_rois.py
```
The predictions should be generated as zip files next to the input images.

For further options, please run `python predict_rois.py -h`.

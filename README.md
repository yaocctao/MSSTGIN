The PyTorch implementation of MSSTGIN model.

## RUNNING ENVIRONMENT

Note that we need to install the right packages to guarantee the model runs according to the file requirements.txt

##### Create Environment

We recommend using conda to create the environment

```bash
conda create --name demo python=3.8
```

Activate the environment (note that after initializing the conda environment variable, restart the terminal for the first time, otherwise it will not be activated)

```bash
conda activate demo
```

Installation of the project environment is required, using this project's requirements.txt for installation

```bash
pip install -r ./requirments.txt
```

## TRAIN & EVALUATE

You can use the command line to train the MSSTGIN model and save the weights under the folder of weights.

```bash
python train.py
```

If you want to evaluate the metrics use the weights, you can use the command below.

```bash
python evaluate.py
```

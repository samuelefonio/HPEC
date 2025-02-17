# Code implementation for the paper "Hyperbolic Prototypical Entailment Cones for Image Classification" 

This is the official repository for the paper "Hyperbolic Prototypical Entailment Cones for Image Classification"

Before running the code, install the requirements.txt 

```
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the packages from requirements.txt
pip install -r requirements.txt
```

The configuration files in configs allow for easy and immediate reproducibility. In case a dataset among cars, cub2011 or aircraft is not downloaded yet, please run the respective python file. The most difficult dataset to get is the cars dataset, a useful link in the python file is provided.

Once the dataset is downloaded, the command to run an experiment is:
```
python main.py -config configs/config.json
```

## Baselines

The configuration files in configs allow for easy and immediate reproducibility. In case a dataset among cars, cub2011 or aircraft is not downloaded yet, please run the respective python file. The most difficult dataset to get is the cars dataset, a useful link in the python file is provided.

Once the dataset is downloaded, the command to run an experiment is:
```
python baselines.py -config baseline_configs/config_HMGP.json -device cuda:0
```

Many configurations are available. Our code reproduces the other basselines as well, except for HBL, which was reproduced with the due alignment from the official repository of the paper https://github.com/MinaGhadimiAtigh/Hyperbolic-Busemann-Learning/tree/master.

# Code implementation for the paper "Hyperbolic Prototypical Entailment Cones for Image Classification" 

This is the official repository for the paper "Hyperbolic Prototypical Entailment Cones for Image Classification"

This repository is released for reproducibility, if any problem occurs please contact us at [omitted for anonimity]

Before running the code, install the requirements.txt 

```
# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install the packages from requirements.txt
pip install -r requirements.txt
```

The configuration files in configs allow for easy and immediate reproducibility. In case a dataset among cars, cub2011 or aircraft is not downloaded yet, please run the respective python file. The most difficult dataset to get is the cars dataset, a useful link in the python file is provided.

Once the dataset is downloaded, the command to run an experiment is:
```
python main.py -config configs/config.json
```


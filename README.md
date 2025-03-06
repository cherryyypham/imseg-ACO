# Ant Colony Optimization Algorithm for Image Segmentation

## PROJECT STRUCTURE
* **algorithm.py**: Contains the ACO algorithm implementation
* **application.py**: Contains functions for image preprocessing and visualization
* **main.py**: Entry point for running the segmentation on an image
* **README.md**: Documentation for the project

## SET-UP INSTRUCTIONS
### Create a virtual environment
```
python -m venv env
```

### Activate the virtual environment on macOS/Linux:
```
source aco_env/bin/activate
```

### Install dependencies
```
pip install -r requirements.txt
```

### Register the kernel with Jupyter
```
python -m ipykernel install --user --name=env --display-name="Python (ACO Segmentation)"
```

## USAGE
Run the main script to segment an image:
```
python main.py
```
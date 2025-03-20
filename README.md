# coral_spawn_counter
Code to count coral larvae in the water column of the larval rearing tanks.


## Installation
- virtual environment to create isolated workspace and maintain separate dependencies from global Python installation
- install conda via miniforge https://github.com/conda-forge/miniforge: `wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"`
- make venv file executable via `chmod +x make_cslics_venv.sh`
- run `./make_cslics_venv.sh` 
- conda environment for cslics should be compiled

## Operation

- `conda activate cslics` to activate virtual environment

## Annotations

- Assisted annotation via Hough Transforms for circular objects in the image, run `python sphere_annotations.py`, which should take existing annotations.xml file and append all circles as bounding boxes for each image
- Save new .xml file as a zip and re-upload to cvat (overwriting previos annotations)
- Check/view/modify annotations in cvat, download when annotations are ready for training


## Data locations
All cslics runs are in Rstore `smb://rstore.qut.edu.au/projects/sef/marine_robotics/dorian/rrap/cslics` 

Current best ultralytics model `/home/java/Java/ultralytics/runs/detect/train - aten_alor_2000/weights/best.pt` trained on 2000 images of aten and alor cslics

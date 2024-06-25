# coral_spawn_counter
Code to count coral spawn (eggs), run on central computer.


## Installation

- install CVAT locally: https://opencv.github.io/cvat/docs/administration/basics/installation/#ubuntu-1804-x86_64amd64
- run `make_cslics_venv.sh` 

- `conda activate cslics`
- navigate to coral_spawn_counter folder
- `pip install -e .` installs as a python package locally so other modules can use this code

- Will also want to install machine-toolbox (0.5.4)
- NOTE, currently using old code and code in these files does NOT work with updated machine-toolbox
- `git clone https://github.com/petercorke/machinevision-toolbox-python.git`
- `cd machinevision-toolbox-python`
- `pip install -e .`

## Operation

- `conda activate cslics` to activate virtual environment
- Open cvat by going to `http://localhost:8080` in Google Chrome (cvat only works in Google Chrome)
- Create project/tasks by creating spawn annotation/label and uploading the relevant images
- Once uploaded to cvat, export the dataset as a .zip file in cvat annotation format (should have `annotations.xml` file)

## Annotations

- Assisted annotation via Hough Transforms for circular objects in the image, run `python sphere_annotations.py`, which should take existing annotations.xml file and append all circles as bounding boxes for each image
- Save new .xml file as a zip and re-upload to cvat (overwriting previos annotations)
- Check/view/modify annotations in cvat, download when annotations are ready for training


## Data locations
Detector fail cases (blank images) `/home/java/Java/data/cslics_failurecases` - 275 images currently, more can be sourced
2023 Dec Alor Tank4 cslics01 and 2023 Dec Alor tank3 cslics06 in `/home/java/Java/data`
other cslic runs are also on the SSD card `/media/java/cslics_ssd`
All cslics runs are in Rstore `smb://rstore.qut.edu.au/projects/sef/marine_robotics/dorian/rrap/cslics` 

Current best ultralytics model `/home/java/Java/ultralytics/runs/detect/train - aten_alor_2000/weights/best.pt` trained on 2000 images of aten and alor cslics

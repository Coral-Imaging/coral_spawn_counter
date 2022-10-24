# coral_spawn_counter
Code to count coral spawn (eggs), run on central computer.


## Installation

- install CVAT locally: https://opencv.github.io/cvat/docs/administration/basics/installation/#ubuntu-1804-x86_64amd64
- run `make_cslics_venc.sh` 

## Operation

- `conda activate cslics` to activate virtual environment
- Open cvat by going to `http://localhost:8080` in Google Chrome (cvat only works in Google Chrome)
- Create project/tasks by creating spawn annotation/label and uploading the relevant images
- Once uploaded to cvat, export the dataset as a .zip file in cvat annotation format (should have `annotations.xml` file)

## Annotations

- Assisted annotation via Hough Transforms for circular objects in the image, run `python sphere_annotations.py`, which should take existing annotations.xml file and append all circles as bounding boxes for each image
- Save new .xml file as a zip and re-upload to cvat (overwriting previos annotations)
- Check/view/modify annotations in cvat, download when annotations are ready for training

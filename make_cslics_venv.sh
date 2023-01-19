# script to automatically make conda environment defined in cslics.yml
mamba env create -f cslics.yml
# conda activate cslics

# in the cslics environment
# pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies

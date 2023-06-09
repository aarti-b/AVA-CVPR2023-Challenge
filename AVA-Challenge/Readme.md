# AVA-Evalai-Challenge

```
    python3 -m venv .env
    source .env/bin/activate
    python3 -m pip install -r requirements.txt
```

```
    cd AVA-Challenge/

```
For pre-processing data ie removing empty labels, converting COCO to YOLO format (bbox) and extracting mask as ```.png``` format, use ```data_processing.ipynb.```

Use ```detectron_fine_tune_training.ipynb``` for training model. the models will be saved in ```segmentation``` folder. Use ```detectron_inference.ipynb``` for inference.

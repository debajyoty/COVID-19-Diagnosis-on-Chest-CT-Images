# Instance Segmentation model for Diagnosis of COVID-19 Cases on Chest CT Images using Mask R-CNN


## DATASET PREPARATION AND SETUP

In order to train the model on our dataset you need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset.

<b>Link to the covid dataset</b>
https://mosmed.ai/datasets/covid19_1110

```Convert dataset into VGG image format for the proposed model```

In order to fit these datasets in our model it needs to be converted into VGG format which contains image and it's metadata in a json file. These converted datasets can be found in the "datasets" folder.

```Requirements to be installed in your sytem in order to run our model```
<ul>
<li>numpy</li>
<li>scipy</li>
<li>Pillow</li>
<li>cython</li>
<li>matplotlib</li>
<li>scikit-image</li>
<li>tensorflow-gpu version 1.15.0</li>
<li>keras version 2.2.5</li>
<li>opencv-python</li>
<li>h5py</li>
<li>imgaug</li>
<li>IPython[all]</li>
</ul>

## MASK R-CNN  
Our model needs MASK-RCNN as a dependency to run which can be downloaded from the given link.

<b>Link to the Mask R-CNN repositiory</b>
https://github.com/matterport/Mask_RCNN

The code is documented and designed to be easy to extend.

## Installation of MASK-RCNN and setup our model
1. Clone the MASK-RCNN repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
4. Now after installation move pre-trained covid weights (mask_rcnn_covid.h5) to the model logs directory and rename the h5 file to (mask_rcnn_balloon.h5) because the code of mask is designed to take h5 file in the following name format. 

```Now since the setup is ready use the following command to train the model```

```bash 
#Run the following command to start training
python3 balloon.py train --dataset="path to dataset" --weights="path to weights"
```

```In our case the code was:```
```bash
!python3 balloon.py train --dataset=/content/drive/My\ Drive/Mask_RCNN-master/balloon_dataset/balloon/ --weights=balloon
```
# Testing the model (Python notebook recommended)

1. Load the model weights

```bash
weights_path = model.find_last() #the will load the last model from the logs directory in our case it's mask_rcnn_covid.h5 which is renamed to mask_rcnn_balloon.h5
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
```

2. select a random image from the test dataset to perform instance segmentation of GGOs

```bash
image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
```
[This blog post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) describes in more detail by taking an example of Instance Segmentation of balloons with Mask R-CNN and TensorFlow.
***To test images in bulk***
make sure the images are in VGG format

```Following steps to be followed for bulk image processing```

1. Place our covid.ipynb file in MASK_RCNN root directory.
2. Place all the images in val directory.
3. Place the mask_rcnn_covid.h5 in model's log directory.
4. Call the "test_bulk_images(image_id)" fuction (Before that make sure you have done all the additional setup of mask-rcc explained above).


<I><p style="color:red;">"If some errors arises it might be due to different filenames of the directories and h5 files, So please make sure you have set then names of the directories in sync with the model"</p></I>

















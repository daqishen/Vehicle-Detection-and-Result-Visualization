# vehicle-detection-and-result-visualization
This model is to detect the vehicle and print the result in a heatmap.
The test model is bascially Faster-RCNN model, mostly trained by (big thanks to) Chenge.
The whole project contains three different part: Fetching data, Processing the image by Faster RCNN model, Visualizing the result in the heatmap.

Instalization: This faster rcnn model repository is based on @endernewton tf-faster-rcnn model. The installation is https://github.com/endernewton/tf-faster-rcnn#installation.
Since this repository is used for windows system, other os has not been test before. Please make sure the prerequisites are well done before.

For each image, the model will detect the vehicles in the image. For example, this is the result of the car detection.

![image](https://github.com/daqishen/vehicle-detection-and-result-visualization/blob/master/data/imgs/result.JPG)

The model contains 5 classes: background, car, bus ,van and others.
The test demo is at tool/iamangry.py
Please store the image at data/demo folder and rename the image in 6 digits.

The next step is to draw the heatmap. The drawing function is bascially use folium library. You need to add geojson file to find the site to draw the map. This demo contains several site information in New York Manhattan. I separate it into several pieces and draw a heatmap as an example:

![image](https://github.com/daqishen/vehicle-detection-and-result-visualization/blob/master/data/imgs/heatmap.JPG)

As a reminder, the model use Tensorflow-GPU (my gpu is GTX1050). So please use an suitable gpu to process the whole model.(cpu model might be in trouble) 




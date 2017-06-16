# Vehicle Detection
In this project, the main goal was to use computer vision and machine learning techniques to identify and track vehicles in images and video feeds.

Specifically, the project consisted of the following steps:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, and append to your HOG feature vector
* Normalize the features and randomize a selection for training and testing
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected

[//]: # (Image References)
[vehicles]: ./output_images/vehicles.png
[non_vehicles]: ./output_images/non_vehicles.png
[hog_examples]: ./output_images/hog_examples.png
[color_hist]: ./output_images/color_hist.png
[spatial_bin]: ./output_images/spatial_bin.png
[search_windows_separate]: ./output_images/search_windows_separate.png
[located_cars]: ./output_images/located_cars.png
[vehicle_final]: ./output_images/vehicle_final.png
[vehicle_final_efficient]: ./output_images/vehicle_final_efficient.png
[heatmap]: ./output_images/heatmap.png
[vehicle_windows]: ./output_images/vehicle_windows.png
[vehicle_heatmap]: ./output_images/vehicle_heatmap.png

## Getting Started
There are many features that we can consider extracting from an image or a video frame. We can first take a look at the images that we have in the training set to get a sense for what might work well.







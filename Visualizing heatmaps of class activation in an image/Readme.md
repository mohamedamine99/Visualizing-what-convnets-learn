<div align="center">
  
# Visualizing heatmaps of class activation in an image using GradCAM 
  
</div>
<p align="center">
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/cat.PNG" width=200>
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/cat_heatmap_overlayed.png" width=200>
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20dog%20video.gif" width=125>
    <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/dog_1_heatmap.gif" width=125>

</p>
Heatmaps are a powerful tool for understanding how a machine learning model makes decisions. In particular, heatmaps of class activation can help us understand which parts of an image the model is using to classify it. One popular method for generating these heatmaps is GradCAM, which stands for Gradient-weighted Class Activation Mapping. With GradCAM, we can generate a heatmap that highlights the regions of an image that are most important for a particular class.

By visualizing these heatmaps, we can gain insight into how our machine learning model is making decisions, which can help us improve our models and build better AI systems.

## Overview 

- `GradCAM_demo_1.ipynb` : This [file](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/GradCAM_demo_1.ipynb) contains code or a simple implementation of the GradCAM algorithm on a single image. The resulting heatmap is then processed and overlaid onto the original image to highlight the parts of the image that the CNN Sequential model is using to make its prediction.

- `Gradcam.ipynb` : This [file](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/Gradcam.ipynb) has code that builds upon the previous `GradCAM_demo_1.ipynb` implementation, adding new functionalities and improvements such as video heatmap overlay and gif creations. In this file we computed the heatmaps for both pre-trained CNN sequential binary classification model and a multi-class (1000) MobilenetV2 model with imagenet weights.


- `gradcam_funcs.py` : This [file](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/gradcam_funcs.py)  encompasses all functions utilized in `Gradcam.ipynb`, with detailed Numpy style documentation for each function.


**What is GradCAM** ?
- Grad-CAM (Gradient-weighted Class Activation Mapping) is a popular technique for visualizing the regions of an image that are important for a neural network's prediction.
It is a technique that uses the gradient of a target class to produce a coarse localization map highlighting the important regions in the input image for predicting 
that class.

**How does GradCAM work** ?  
- The basic idea behind Grad-CAM is to take the output feature maps of the last convolutional layer in the network
and weigh them by the gradients of the target class with respect to the feature maps. 
This produces a weighted sum of the feature maps, where each feature map is weighted by the importance of its activations for predicting the target class. 
This weighted sum is then passed through a ReLU activation function to obtain the final Grad-CAM visualization.

**How does GradCAM help interpret Convolutional Neural Network Decisions** ?  
- The Grad-CAM visualization can be superimposed onto the original image, highlighting the regions of the image that are most important for the predicted class. 
This can help provide insight into how the network is making its predictions and can be useful for interpreting and explaining the decisions made by the neural network.

## Results :


<div align="center">

Heatmaps of class activation with a CNN Sequential Model binary classifier overlayed on image files
<details>
  <summary>Click to view results</summary>

<div align="center">  
<table style="margin: 0 auto; border-style: none; width:100%">
  <tr>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/cat.PNG" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/dog.PNG" width=170></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/2%20cats.PNG" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/2%20dogs.PNG" width=170></td>
  </tr>

  <tr>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/cat_heatmap_overlayed.png" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/dog_heatmap_overlayed.png" width=170></td>  
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/2%20cats_heatmap_overlayed.png" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/2%20dogs_heatmap_overlayed.png" width=170></td>
  </tr>


  
</table>
</div>
</div>

</details>

<div align="center">
Heatmaps of class activation with a CNN Sequential Model binary classifier overlayed on video files
<details>
  <summary>Click to view results</summary>
  
<div align="center">  
<table style="margin: 0 auto; border-style: none; width:100%">
  <tr>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20cat%20video%202.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20cat%20video.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20dog%20video%202.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20dog%20video.gif" width=140></td>
  </tr>

  <tr>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/cat_2_heatmap.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/cat_3_heatmap.gif" width=140></td> 
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/dog_4_heatmap.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/dog_1_heatmap.gif" width=140></td>
    
  </tr>
  
</table>
</div>
</div>

</details>

<div align="center">
Heatmaps of class activation with a multi-class MoblinetV2 classifier overlayed on video files

<details>
  <summary>Click to view results</summary>
  


<div align="center">  
<table style="margin: 0 auto; border-style: none; width:100%">
  <tr>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20cat%20video%202.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20cat%20video.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20dog%20video%202.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/unprocessed%20gifs/input%20dog%20video.gif" width=140></td>
  </tr>

  <tr>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/mnet_cat_2_heatmap.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/mnet_cat_3_heatmap.gif" width=140></td> 
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/mnet_dog_4_heatmap.gif" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/gifs/mnet_dog_1_heatmap.gif" width=140></td>
    
  </tr>
  
</table>
</div>
</div>

</details>

<div align="center">
  
# Visualizing heatmaps of class activation in an image using GradCAM 
  
</div>
<p align="center">
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/cat.PNG" width=200>
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/Visualizing%20heatmaps%20of%20class%20activation%20in%20an%20image/results/cat_heatmap_overlayed.png" width=200>

</p>

Grad-CAM (Gradient-weighted Class Activation Mapping) is a popular technique for visualizing the regions of an image that are important for a neural network's prediction.
It is a technique that uses the gradient of a target class to produce a coarse localization map highlighting the important regions in the input image for predicting 
that class.

The basic idea behind Grad-CAM is to take the output feature maps of the last convolutional layer in the network
and weigh them by the gradients of the target class with respect to the feature maps. 
This produces a weighted sum of the feature maps, where each feature map is weighted by the importance of its activations for predicting the target class. 
This weighted sum is then passed through a ReLU activation function to obtain the final Grad-CAM visualization.

The Grad-CAM visualization can be superimposed onto the original image, highlighting the regions of the image that are most important for the predicted class. 
This can help provide insight into how the network is making its predictions and can be useful for interpreting and explaining the decisions made by the neural network.






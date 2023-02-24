# Visualizing intermediate layers activations
To gain a deeper understanding of a CNN's inner workings, one can use visualizations of intermediate activations to observe how the image is transformed by each convolutional layer. This approach can help to identify which features of the input image are being extracted and processed by the network. It can also help to identify any limitations or issues with the network's performance.

Visualizing intermediate activations consists of displaying the feature maps that are output by various convolution and pooling layers in a network, given a certain input (the output of a layer is often called its activation, the output of the activation function).These feature maps, also known as activations, illustrate how inputs are broken down into different filters learned by the network. The feature maps contain three dimensions - width, height, and depth - and since each channel represents relatively independent features, they are best visualized by plotting the contents of each channel as a separate 2D image.
 

* The model used here is a **CNN image classifer** for cats and dogs images with 96 % accuracy. 
  You can find the saved model in .hdf5 format [here](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/models/CNN%20cats%20vs%20dogs%20model.hdf5)
* The testing images that we'll be using can be found [here](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/tree/main/test%20images).
<details>
  <summary>Click to view test images</summary>

<p align="center">
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/dog.PNG" width=150>
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/cat.PNG" width=150>
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/2%20dogs.PNG" width=150>
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/2%20cats.PNG" width=150>
</p>
</details>

* The resulting images and animations can be found [here](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/tree/main/visualizing%20intermediate%20layers%20activations/results).  The results folder contains 3 other folders:
    - [activations](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/tree/main/visualizing%20intermediate%20layers%20activations/results/activations) : contains all the activations of the different layers in the model in a grid format for each of the 4 test images in separate folders.
  - [activations_2](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/tree/main/visualizing%20intermediate%20layers%20activations/results/activations_2) : contains all the activations of the different layers in the model with a single channel per image ,for each of the 4 test images in separate folders.

  - [activations GIFs](https://github.com/mohamedamine99/Visualizing-what-convnets-learn/tree/main/visualizing%20intermediate%20layers%20activations/results/activations%20GIFs) : contains all the animations of activations of the different layers in the model for each of the 4 test images in separate folders.
 
### Full activations channels displayed as a grid
<p align="center">
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations%20GIFs/cat%20activations.gif" width=500>
</p>

### Conv layers activations channels displayed individually 

<div align="center">  
<table style="margin: 0 auto; border-style: none; width:100%">
  <tr>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/cat.PNG" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/dog.PNG" width=170></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/2%20cats.PNG" width=140></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/2%20dogs.PNG" width=170></td>
  </tr>
  <tr>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations%20GIFs/cat%20activations%202.gif" width=200></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations%20GIFs/dog%20activations%202.gif" width=200></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations%20GIFs/two%20cats%20activations%202.gif" width=200></td>
    <td><img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations%20GIFs/two%20dogs%20activations%202.gif" width=200></td>
  </tr>
</table>
</div>

### Interpreting results:

* The first layer, with filters of size (3x3) applied on an image of size (128x128) acts as a collection of various edge detectors. Therefore, the activations retain almost all of the information present in the initial picture.
<p align="center">
<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/cat.PNG" width=150>
<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/cat/00--001.png" width=150>
<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/cat/00--025.png" width=150>
 <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/dog.PNG" width=150>
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/dog/00--001.png" width=150>
   <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/dog/00--025.png" width=150>
</p>

* The higher we go, the more the activations become abstract and less visually interpretable. They begin to encode higher-level concepts such as “cat ear” and
“cat eye.” Higher presentations carry increasingly less information about the visual contents of the image, and increasingly more information related to the
class of the image.

<p align="center">
<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/test%20images/cat.PNG" width=150>
<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/cat/00--001.png" width=150>
<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/cat/05--025.png" width=150>

<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/cat/09--012.png" width=150>

<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/cat/09--025.png" width=150>

<img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/results/activations_2/cat/13--050.png" width=150>
</p>

## How it works :

In order to extract the activation of the different layers of the model we need to create a Keras model that takes batches of images as input, and outputs the activations of all convolution and pooling layers. To do this, we would need to recreate our CNN classification model into a new modified version that has the same input but multiple outputs. the new model's outputs are the outputs of the different layers of the original model. we’ll use the Keras class `Model`. A model is instantiated using two arguments: an input tensor (or list of input tensors) and an output tensor(or list of output tensors).The new model (named `activations_model` in the code) mapping the specified inputs (img tensor or list of multiple tensors) to the specified outputs.

<p align="center">
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/Readme%20imgs/original%20model.png" width=800>
  <img src="https://github.com/mohamedamine99/Visualizing-what-convnets-learn/blob/main/visualizing%20intermediate%20layers%20activations/Readme%20imgs/activations%20model.png" width=800>

</p>



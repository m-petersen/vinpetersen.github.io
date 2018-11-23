---
layout: post
published: false
title: A Guided Tour Through a Convolutional Neural Network - Part 1
---
### Introduction

This series of posts wants to introduce in the topic of convolutional neural networks (CNN). By following the path of data flowing through the network, the goal is to establish an intuitive understanding of the inner workings of those algorithms. Thereby, this work is meant to be especially valuable for readers from a non-mathematical, non-computer-science background just starting out on the subject. My inspiration to put the things I've learned about CNN into this format is drawn from the Fastai live course I am currently attending. The knowledge I'm sharing here comes mostly from reading various materials and attending different courses for the purpose of writing a thesis in a student research project. A major part of these posts is taken from this thesis I finished recently.

### The anatomy of a CNN

![Figure 1. Example net]({{site.baseurl}}/img/example_net.tif)
*Figure 1. Example net*

The tour through the network follows an exemplary network illustrated in Figure 1. It is inspired by LeNet, (LeCun et al., 1998) the first CNN architecture. While the specific architecture differs between CNN, they are characterised by stereotypical elements covered by this example network. At this point it is important to mention, that the figure and the immediately following paragraphs serve to create a frame for the walkthrough. Therefore, all the terms mentioned will be explained in greater detail later.

As can be seen in Figure 1, the network is subdivided into a forward and backward pass. During the forward pass, the data goes through distinct layers (Fig. 1: coloured arrows). The input image (Fig. 1: empty rectangle), represented by a two-dimensional array, is provided to the name-giving convolutional layer. This layer identifies contours and shapes on the input data and outputs a stack of feature maps (Fig. 1: vertically striped rectangles). A max pooling layer succeeds the convolutional layer, eliminating insignificant parts and thereby shrinking the data. The same constellation of both layers follows thereupon.

Thereafter the data passes a flattening operation, converting the image data into a vector, leading into the fully-connected layers (Fig. 1: horizontally striped rectangles). Those identify a class-specific pattern in this vector and, based on that, predict a class membership of the input data.

Hitherto, no learning has occurred. That is accomplished in the backward pass. At first, a classification error is quantified by a loss function. Based on the result of the loss function, parameters in the aforementioned layers are optimised by backpropagation, which finalises one iteration through the network. A noticeably higher count of iterations, i.e. input of more training sample images, is necessary to achieve proper classification results. The point at which the complete training dataset passed through the model is called epoch.

### Convolution, Rectified Linear Units and Max Pooling

![Figure 2. Convolution operation]({{site.baseurl}}/img/convolution_.png)
*Figure 2. Convolution operation*

The term ‘convolution’ describes a specific type of matrix calculation where special purpose matrices called filters are applied to the input image. The process is illustrated in *Figure 2*. A filter is typically a smaller matrix, and in a convolution operation, it is placed on a subpart of the image (*Fig. 2*: 3x3 coloured frame). An elementwise multiplication of each couple of corresponding values and a subsequent sum over all products takes place, resulting in a single output value. After that, the filter is moved in a sliding window fashion over the image performing the above computation on every fitting subset of the input image. The resulting output values are collected in an output matrix called feature map, whereby the location of the output value in the feature map (Fig. 2: coloured 1x1 frame) corresponds to the location of the input image subset involved in the calculation. Typically, the feature map is of smaller size than the input image. During a convolution operation, mostly multiple individual filters are applied to the input image, resulting in a feature map for each filter. In other words, a convolutional layer outputs a stack of feature maps corresponding to its filter count.

A filter can be considered as a specialised contour detector, and the resulting feature map reports detection locations. If a filter is placed on an image subpart containing an edge – defined by a considerable intensity contrast between adjacent values – it is specialised for, it will translate its finding into a high value in the feature map. In other words, a high feature map value signifies a contour detection in the input image at the specific location. The process is also illustrated in Figure 2: if the filter reaches the subparts marked by the yellow and green frame it identifies the underlying contour. Consequently, feature maps can be seen as images as well and visualised accordingly.

![]({{site.baseurl}}/img/zeiler_fergus.tif)
*Figure 3. Filters after learning; from Zeiler and Fergus, 2014*

Filter values are weights and thus among the parameters that are learned. They are continuously optimised during the backward pass while further data passes through the network. By that an adjustment process is achieved: the filters are learning to recognise specific elements available in the input images and can be visualised as pictures themselves. Whereas filters in earlier layers (left side Figure 3) will learn basal picture elements like contours, filters in later layers (right side Figure 3) will join upstream features and by that learn more complex constructs like for example eyes or even faces (Zeiler and Fergus, 2014). Thus, filters can be cautiously compared with receptive fields of visual cortex neurons.

A function called rectified linear unit (ReLU) is applied to the output feature maps. Under the intimidating name hides a simple thresholding step: all values below zero are zeroed.

The thresholded feature maps are handed over to a max pooling layer. Here, a simpler type of matrix calculation happens, though similar to convolution. A filter is placed on feature map subsets, again in sliding-window fashion, and extracts the highest value of the subsets, keeping them in a condensed output feature map. The purpose is dropping superfluous data: values not signifying any contour detection are crossed-out, whereas the spatial information is roughly kept. This results in lower computation costs.
### 

Enter text in [Markdown](http://daringfireball.net/projects/markdown/). Use the toolbar above, or click the **?** button for formatting help.

---
layout: post
published: false
title: A Guided Tour Through a Convolutional Neural Network - Part 1
---
### Introduction

This series of posts wants to introduce in the topic of convolutional neural networks (CNN). By following the path of data flowing through the network, the goal is to establish an intuitive understanding of the inner workings of those algorithms. Thereby, this work is meant to be especially valuable for readers from a non-mathematical, non-computer-science background just starting out on the subject. My inspiration to put the things I've learned about CNN into this format is drawn from the Fastai live course I am currently attending. The knowledge I'm sharing here comes mostly from reading various materials and attending different courses for the purpose of writing a thesis in a student research project. A major part of these posts is taken from this thesis I finished recently.

### The anatomy of a CNN

![example_net.tif]({{site.baseurl}}/img/example_net.tif)

The tour through the network follows an exemplary network illustrated in Figure 1. It is inspired by LeNet, (LeCun et al., 1998) the first CNN architecture. While the specific architecture differs between CNN, they are characterised by stereotypical elements covered by this example network. At this point it is important to mention, that the figure and the immediately following paragraphs serve to create a frame for the walkthrough. Therefore, all the terms mentioned will be explained in greater detail later.
As can be seen in Figure 1, the network is subdivided into a forward and backward pass. During the forward pass, the data goes through distinct layers (Fig. 1: coloured arrows). The input image (Fig. 1: empty rectangle), represented by a two-dimensional array, is provided to the name-giving convolutional layer. This layer identifies contours and shapes on the input data and outputs a stack of feature maps (Fig. 1: vertically striped rectangles). A max pooling layer succeeds the convolutional layer, eliminating insignificant parts and thereby shrinking the data. The same constellation of both layers follows thereupon.
Thereafter the data passes a flattening operation, converting the image data into a vector, leading into the fully-connected layers (Fig. 1: horizontally striped rectangles). Those identify a class-specific pattern in this vector and, based on that, predict a class membership of the input data. 
Hitherto, no learning has occurred. That is accomplished in the backward pass. At first, a classification error is quantified by a loss function. Based on the result of the loss function, parameters in the aforementioned layers are optimised by backpropagation, which finalises one iteration through the network. A noticeably higher count of iterations, i.e. input of more training sample images, is necessary to achieve proper classification results. The point at which the complete training dataset passed through the model is called epoch.




## A New Post

Enter text in [Markdown](http://daringfireball.net/projects/markdown/). Use the toolbar above, or click the **?** button for formatting help.

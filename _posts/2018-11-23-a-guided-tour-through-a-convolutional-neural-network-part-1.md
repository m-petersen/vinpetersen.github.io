---
layout: post
published: true
title: A Guided Tour Through a Convolutional Neural Network - Part 1
subtitle: 'Introduction, Convolution and Pooling'
date: '2018-11-23'
---
### Introduction

This series of posts aims to introduce to the topic of convolutional neural networks (CNN) in a comprehensive and concise manner. By following the path of data flowing through the network, the goal is to establish an intuitive understanding of the inner workings of those algorithms. Thereby, this work is meant to be especially valuable for readers from a non-mathematical, non-computer-science background just starting out on the subject.

My inspiration to put the things I've learned about CNN into this format comes from the Fast.ai course 'Practical Deep Learning For Coders v3', which I'm currently attending, where the instructor Jeremy Howard encouraged us to blog about the things we have learned. The knowledge I'm sharing here comes from reading various materials and attending different courses for the purpose of writing a thesis in a student research project with a medical imaging focus. A major part of these posts is taken from this thesis I finished recently.

Before the journey begins, some precautions have to be taken to put convolutional neural networks (CNN) in perspective. Machine learning (ML) is the subfield of computer science addressing problems through algorithms with learning capabilities. The ‘learning’ happens via autonomous optimisation of internal components, called parameters. Problems suitable for ML are two forms of predictions: regression – the prediction of continuous values – and classification – the subdivision of objects into distinct groups by class membership prediction.

Deep learning (DL) in turn is a subfield of ML, distinguished by applying algorithms with a specific architecture, called neural networks, to machine learning problems. This architecture is inspired by neural networks in nature and consists – somewhat simplified – of neurons representing computational units and edges connecting different neurons as well as ensuring the flow of information. There are different types of neural networks: artificial neural networks (ANN), specialised on working with tabular data; recurrent neural networks (RNN) for time-series data like speech and convolutional neural networks (CNN), especially applicable to image data. Armed with this basics we can start to investigate the latter type.

### The anatomy of a CNN

![example_net.png]({{site.baseurl}}/img/example_net.png)
*Figure 1. Example net*

The tour through the network follows an exemplary network illustrated in *Figure 1*. It is inspired by LeNet, (LeCun et al., 1998) the first CNN architecture. While the specific architecture differs between CNN, they are characterised by stereotypical elements covered by this example network. The algorithm is designed to solve a binary classification problem such as the distinction between cats and dogs.

At this point, it is important to mention, that the figure and the immediately following paragraphs serve to create a frame for the walkthrough. While a deeper understanding will gradually be developed in the coming parts, *Fig. 1* will serve as a reference to facilitate orientation for the reader. Hence I want you to endure some minutes 

-----Which means each computation succesively applied to the data mentioned here will be discussed more in-depth later This frame shall serve as a reference  Therefore, all the terms mentioned will be explained in greater detail later and the reader is encouraged to endure some questions.-----

As can be seen in *Figure 1*, the network is subdivided into a forward and backward pass. During the forward pass, the data goes through distinct layers (*Fig. 1*: coloured arrows). The input image (Fig. 1: left empty rectangle), represented by a two-dimensional array, is provided to the name-giving convolutional layer. This layer identifies contours and shapes on the input data and outputs a stack of feature maps (Fig. 1: vertically striped rectangles). A max pooling layer succeeds the convolutional layer, eliminating insignificant parts and thereby shrinking the data. The same constellation of both layers follows thereupon.

Thereafter the data passes a flattening operation, converting the image data into a vector, leading into the fully-connected layers (*Fig. 1*: horizontally striped rectangles). These layers identify a class-specific pattern in this vector and, based on that, predict a class membership of the input data.

Hitherto, no learning has occurred. That is accomplished in the backward pass. At first, a classification error is quantified by a loss function. Based on the result of the loss function, parameters in the aforementioned layers are optimised by backpropagation and gradient descent, which finalises one iteration through the network. A noticeably higher count of iterations, i.e. input of more training sample images, is necessary to achieve proper classification results. The point at which the complete training dataset passed through the model is called epoch.

### Convolution, Rectified Linear Units and Max Pooling

![convolution.png]({{site.baseurl}}/img/convolution.png)

*Figure 2 Convolution operation*

The term ‘convolution’ describes a specific type of matrix calculation where special purpose matrices called filters are applied to the input image. The process is illustrated in *Figure 2*. A filter (*Fig. 2*: dark 3x3 matrix) is typically a smaller matrix, and in a convolution operation, it is placed on a subpart of the image (*Fig. 2*: 3x3 cyan image subset). An elementwise multiplication of each couple of corresponding values and a subsequent sum over all products takes place, resulting in a single output value. In other words, the top left value of the image subset is multiplied with the top left value of the filter, top middle with the corresponding top middle (...) and in the end all products are summed. After that, the filter is moved in a sliding window fashion over the image performing the above computation on every fitting subset of the input image. The resulting output values are collected in an output matrix called feature map, whereby the location of the output value in the feature map (*Fig. 2*: cyan top middle value) corresponds to the location of the input image subset involved in the calculation. 

How exactly the filter moves over the image is determined by stride and padding. A stride of one describes moving the filter by one pixel per convolution operation resulting in a smaller feature map (*Fig. 2*: stride s = 1). The size reduction of the feature map can be counteracted by adding zero pixels to the outer boundary of the image, called padding (*Fig. 2*: padding = 0). During a convolution operation, mostly multiple individual filters are applied to the input image, resulting in a feature map for each filter (*Fig. 2*: multiple silhouttes behind filter and feature map). In other words, a convolutional layer outputs a stack of feature maps corresponding to its filter count.

![convolution_.png]({{site.baseurl}}/img/convolution_.png)
*Figure 23 Convolution operation*

A filter can be considered as a specialised contour detector, and the resulting feature map reports detection locations. If a filter is placed on an image subpart containing an edge – defined by a considerable intensity contrast between adjacent values – it is specialised for, it will translate its finding into a high value in the feature map. In other words, a high feature map value signifies a contour detection in the input image at the specific location. The process is illustrated in *Figure 3*: if the filter reaches the subparts marked by the yellow and green frame it identifies the underlying contour. Consequently, feature maps can be seen as images as well and visualised accordingly.

![zf.png]({{site.baseurl}}/img/zf.png)
*Figure 4. Filters after learning; from Zeiler and Fergus, 2014*

Filter values are weights and thus among the parameters that are learned. They are continuously optimised during the backward pass while further data passes through the network. By that an adjustment process is achieved: the filters are learning to recognise specific elements available in the input images and can be visualised as pictures themselves. Whereas filters in earlier layers (*Fig. 4*: left) will learn basal picture elements like contours, filters in later layers (*Figure 4*: right) will join upstream features and by that learn more complex constructs like for example eyes or even faces (Zeiler and Fergus, 2014). Thus, filters can be cautiously compared with receptive fields of visual cortex neurons.

![relu.png]({{site.baseurl}}/img/relu.png)
*Figure 5. Rectified linear unit (ReLU)*

A function called rectified linear unit (ReLU) is applied to the output feature maps (*Fig. 5*). Under the intimidating name hides a simple thresholding step: all values below zero are zeroed.

![maxpool.png]({{site.baseurl}}/img/maxpool.png)
*Figure 6. Max Pooling*

The thresholded feature maps are handed over to a max pooling layer (*Fig. 6*). Here, a simpler type of matrix calculation happens, though similar to convolution. A filter is placed on feature map subsets, again in sliding-window fashion, and extracts the highest value of the subsets, keeping them in a condensed output feature map. The purpose is dropping superfluous data: values not signifying any contour detection are crossed-out, whereas the spatial information is roughly kept. This results in lower computation costs.

This concludes part 1 of the series and thereby the first part of the forward pass. In the second part we will carry on with the fully-connected layers. If you liked that post consider sharing. I'd be grateful.

BRAIN CONTRAST FEATURE MAP

### Inspiration and references

* Deep Learning For Coders by Jeremy Howard and Fast.ai: https://course.fast.ai/
* Neural Networks and Deep Learning by Andrew Ng and deeplearning.ai: https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning
* LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., 1998. Gradient-Based Learning Applied to Document Recognition.
* Zeiler, M.D., Fergus, R., 2014. Visualizing and Understanding Convolutional Networks
* A very good post about the content discussed is made by [Ujjwal Karn](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/). He provides multiple animations for better comprehension. Also, it can be very helpful to hear another explanation of the same topic.

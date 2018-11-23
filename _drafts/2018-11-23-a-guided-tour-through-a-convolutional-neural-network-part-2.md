---
layout: post
published: false
title: A Guided Tour Through a Convolutional Neural Network - Part 2
---
### Flattening, the Fully-connected Layer and the Single Neuron

As mentioned in part 1 of the series, there are multiple subtypes of neural networks besides CNN. As one of those subtypes, artificial neural networks (ANN) find widespread application in prediction based on tabular data. Their architecture is also harnessed in the fully-connected (FC) layers of CNN, which forms the final section of the forward pass and results in a prediction.

![neurons_hd.png]({{site.baseurl}}/img/neurons_hd.png)
*Figure 4. Exemplary fully-connected layer*

Prior to the FC layers, the data gets flattened: initially being in matrix form, a row-wise rearrangement of feature map values results in one single vector. Now, the values representing detections are stacked in vector form. FC layers serve the purpose of recognising a class membership specific pattern in the vector. Based on that, a prediction will be made. In the next examples following denotation has been used: scalars are denoted by thin letters (a), vectors by bold (a) and matrices by bold, capital letters (W). A capital T marks a vector or matrix transpose (aT). Superscript indices are referring to the layer affiliation, subscript indices to the localisation in a vector or matrix respectively: scalar a_i^((l)) belongs to layer l and refers to the i-th element of vector a^((l)); Scalar w_(i,j)^((l)) is located at the intersection of the i-th row and j-th column of matrix W^((l)); vector w_(i,)^((l)) refers to the i-th row of matrix W^((l)).

The exact architecture of the FC layers is a directed acyclical graph and thereby resembles a biological neural network. It consists of individual computation units, the neurons, and the connections between them representing weights. The macroscopical structure is divided into distinct layers and exemplarily illustrated in Figure 4: the input layer a^((0)) consists of the input vector and contains as many neurons as the input vector has elements. Its purpose is less computing than distributing incoming values to all neurons of the next layer. The output layer a^((2)) consists of as many neurons as there are classes to predict. Both, input and output layer, are connected by one or multiple hidden layers, here only a^((1)), mainly carrying out the computations the final prediction is based on. Weights or respectively connections are collected in W^((1)) and W^((2)), i.e. are assigned to the layer following them. Noticeably, every neuron is connected with every other neuron of the neighbouring layers, which explains the name choice of ‘fully-connected’ for the layers. Why layers and weights are denoted as vectors and matrices will become clearer while looking at the inner workings of a single neuron.

![single_neuron.tif]({{site.baseurl}}/img/single_neuron.tif)
*Figure 5. Single neuron*

The close-up of the single neuron a_2^((1)) is depicted in Figure 5. The input values of the neuron are the elements of the flattened input vector a^((0)). A neuron in a later layer would operate on the outputs of the antecedent layer. To every input value, an individual weight is assigned. Those weights are as well collected in a vector w_2^((1)), which is the second row of weight matrix W^((1)). This points to the fact that every row of weights in W^((1)) belongs to a neuron in the subsequent layer, while every column to a neuron in the antecedent one.

An elementwise vector multiplication with a subsequent sum over all products is applied. In other words, every input value is multiplied by its assigned weight and subsequently, all products are summed to achieve a scalar. This scalar is summed with another scalar parameter, bias b. Finally, the sum is provided to an activation function. While in the output layer it typically is a sigmoid function, here it is another ReLU. Thus, the subzero values are zeroed. The result a_2^((1)), the so-called activation, is the output of the neuron and at the same time the symbol it is referenced by. Consequently, the layer a^((1)) consists of the outputs it yields collected in a vector and so it is referenced as a vector.

Overall, this computational integration procedure applies to every non-input neuron of the FC layer. Once all the data went through the FC layer, the outputs of the output layer neurons represent interpretable scores for the prediction. The network predicts the class with the highest score value. This finalises the forward pass.

As mentioned before, the FC layer identifies patterns in the input vector which are characteristic for a specific class. In the learning process weights and biases adapt for this task. The goal is to weight the values in a way that increases the prediction score of the true class. This requires a comparison of prediction and true label. How this is achieved will be the subject of the next chapter.

Enter text in [Markdown](http://daringfireball.net/projects/markdown/). Use the toolbar above, or click the **?** button for formatting help.
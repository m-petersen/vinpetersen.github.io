---
layout: post
published: false
title: A Guided Tour Through a Convolutional Neural Network - Part 4
subtitle: The Maths Behind Backpropagation
---
This post is part of a series about the inner workings of convolutional neural networks. In order to be best prepared for the following topics, especially by knowing the denotation, I recommend to deal with parts [1](https://vinpetersen.github.io/2018-11-23-a-guided-tour-through-a-convolutional-neural-network-part-1/), [2](https://vinpetersen.github.io/2018-11-25-a-guided-tour-through-a-convolutional-neural-network-part-2/) and [3](https://vinpetersen.github.io/2019-11-29-a-guided-tour-through-a-convolutional-neural-network-part-3/) before continuing.

To conquer backpropagation in a novice-friendly manner, before we start this quite math heavy part, let's take a look at an computational graph beforehand.

A juxtaposition of mathematical functions can be visualised as a computational graph. *Figure 14* illustrates that process and here variables and functions are shown in rectangles while inputs and function results are shown in circles. An exemplary input of $$a=3$$, $$b=2$$ and $$c=4$$ results in $$J=20$$ as an output.

The parallels to a FC layer appear obvious, since it's also simply a concatenation of functions in computational graph format. The way resulting in the output can be considered as the forward pass (*Fig. 14: grey elements*) of the computational graph. And likewise in common is the possibility to compute the derivative of the last function with respect to an arbitrary value in the graph, which would correspond to the backward pass (*Fig. 14: blue elements*) of a CNN.

Since the computations of the graph are slightly simpler, it shall serve here as an example for a complete computation of the backward pass just as it would appear in a CNN. If simple derivation rules aren't present, I suggest to learn or refresh them before reading further. My go to resource is the course by [Samuel J. Cooper and colleagues](https://www.coursera.org/learn/multivariate-calculus-machine-learning).

Now, let's say we aim to compute $$\frac{∂J}{∂a}$$. To achieve that we start with computing the derivatives of all the nested subfunctions reaching from input $$a$$ to output $$J$$. By application of the common derivation rules it is a simple process: $$\frac{dJ}{dv}=2$$, $$\frac{∂v}{∂u}=1$$ and $$\frac{∂u}{∂a}=b$$. After that, all these derivatives are chained together which yields $$\frac{∂J}{∂a}$$, so $$\frac{∂J}{∂a}=\frac{dJ}{dv}\frac{dv}{du}\frac{du}{da}=2∗1∗2=4$$. Now we know that the gradient of function $$J$$ at point $$a=3$$ is positive and equals 4 and consequently that we should minimise $$a$$ if we would like to minimise $$J$$. 

This whole principle is called chain rule and is also utilised in backpropagation in the absolutely same manner (*Figure 15*: blue elements), even though with other functions and values. The functions in the fully connected layers are traced back by computing their derivatives and then concatenating those leading to the parameter of interest via the chain rule, which results in the derivative of the cross-entropy with respect to the parameter we backpropagated to.

The computations result in the following equations for weights and biases.

$$\frac{∂C}{∂w_{ij}^{l}}=a_{j}^{l-1}\frac{∂C}{∂z_{i}^{l}}$$

To compute the derivative of $$C$$ with respect to a weight, the activation of the neuron the weight is coming from is multiplied by $$\frac{∂C}{∂z_{i}}$$, the derivative of C with respect to the $$z$$ the weight is leading to, (Figure 16). $$\frac{∂C}{∂z_{i}}$$ is also called the error of a neuron.

$$\frac{∂C}{∂b_{i}^{l}}=\frac{∂C}{∂z_{i}^{l}}$$

The derivative regarding a bias is simply the error of the same neuron.

Of course the derivatives for any parameter to be learned are computed. 

Thereby, the name backpropagation gets comprehensible: by applying the chain rule a backward movement through the network is performed.










It's easy to see that a

---
layout: post
published: false
title: A Guided Tour Through a Convolutional Neural Network - Part 4
subtitle: The Maths Behind Backpropagation
---
This post is part of a series about the inner workings of convolutional neural networks. In order to be best prepared for the following topics, especially by knowing the denotation and how convolution operations work, I recommend to deal with parts [1](https://vinpetersen.github.io/2018-11-23-a-guided-tour-through-a-convolutional-neural-network-part-1/), [2](https://vinpetersen.github.io/2018-11-25-a-guided-tour-through-a-convolutional-neural-network-part-2/) and [3](https://vinpetersen.github.io/2019-11-29-a-guided-tour-through-a-convolutional-neural-network-part-3/) before continuing.

### Backpropagation in Detail

To conquer backpropagation in a novice-friendly manner, before we start this quite math heavy part, let's take a look at an computational graph beforehand.

![comp_graph.png]({{site.baseurl}}/img/computational_graph.png)
*Figure 14. Exemplary computational graph*

A juxtaposition of mathematical functions can be visualised as a computational graph. *Figure 14* illustrates that process and here variables and functions are shown in rectangles while inputs and function results are shown in circles. An exemplary input of $$a=3$$, $$b=2$$ and $$c=4$$ results in $$J=20$$ as an output.

The parallels to a FC layer appear obvious, since it's also simply a concatenation of functions visualised in computational graph format. This concatenation resulting in the output $J$ can be considered as the forward pass (*Fig. 14: grey elements*) of the computational graph. And likewise in common is the possibility to compute the derivative of the last function $J(v)$ with respect to an arbitrary value in the graph, which would correspond to the backward pass (*Fig. 14: blue elements*) of the FC layers.

Since the computations of the graph are slightly simpler, it shall serve here as an example for a complete computation of the backward pass just as it would appear in a CNN. Simple derivation rules are necessary to comprehend these computations and I suggest to learn or refresh them before reading further. My go to resource is the course by [Samuel J. Cooper and colleagues](https://www.coursera.org/learn/multivariate-calculus-machine-learning).

Now, let's say we aim to compute $$\frac{∂J}{∂a}$$. To achieve that we start with computing the derivatives of all the nested subfunctions reaching from output $$J$$ to input $$a$$. By application of the common derivation rules it is a simple process: $$\frac{dJ}{dv}=2$$, $$\frac{∂v}{∂u}=1$$ and $$\frac{∂u}{∂a}=b$$. After that, all these derivatives are chained together which yields $$\frac{∂J}{∂a}$$, so in case of the examplified inputs $$\frac{∂J}{∂a}=\frac{dJ}{dv}\frac{dv}{du}\frac{du}{da}=2∗1∗2=4$$. Now we know that the gradient of function $$J$$ at point $$a=3$$ is positive and equals 4 and consequently that we should minimise $$a$$ if we would like to minimise $$J$$. 

![backprop_en_detail.png]({{site.baseurl}}/img/backprop_detailed.png)
*Figure 15: Backpropagation in detail*

This whole principle is called chain rule and is also utilised in backpropagation in the absolutely same manner (*Figure 15*: blue elements), even though with other functions and values. The functions in the fully connected layers are traced back by computing their derivatives and then concatenating those leading to the parameter of interest via the chain rule, which results in the derivative of the cross-entropy with respect to the parameter we backpropagated to.

The computations result in the following equations for weights and biases.

$$\frac{∂C}{∂w_{ij}^{l}}=a_{j}^{l-1}\frac{∂C}{∂z_{i}^{l}}$$

![weight_dev.png]({{site.baseurl}}/img/weight_dev.png)
*Figure 16: Computation of cross-entropy loss with respect to an arbitrary weight*

To compute the derivative of $$C$$ with respect to a weight, the activation of the neuron the weight is coming from is multiplied by $$\frac{∂C}{∂z_{i}}$$, i.e. the derivative of C with respect to the $$z$$ the weight is leading to, (Figure 16). $$\frac{∂C}{∂z_{i}}$$ is also called the error of a neuron (*Fig. 16*).

$$\frac{∂C}{∂b_{i}^{l}}=\frac{∂C}{∂z_{i}^{l}}$$

The derivative regarding a bias is simply the error of the same neuron.

Of course the derivatives for any parameter to be learned are computed. After the computations, the derivatives are utilisied by gradient descent to optimise the weights as discussed in part [3](https://vinpetersen.github.io/2019-11-29-a-guided-tour-through-a-convolutional-neural-network-part-3/).

To move furtherly backwards through the network $$\frac{∂C}{∂z^{l}}$$, i.e. the vector containing all errors of layer $l$, is computed via: 
$$\frac{∂C}{∂z^{l}}=(\boldsymbol{(∂w_{i,j}^{l+1})^{T}})\frac{∂C}{∂z^{l+1}})g'(z^{l})$$

![backprop_detailed2.png]({{site.baseurl}}/img/backprop_detailed2.png)
*Figure 17. Moving the error backwards through the network*

In the equation, the backward movement is retraced. From the initial $\boldsymbol{\frac{∂C}{∂z^{l+1}}}$ a backward step is taken by mutliplying by $\boldsymbol{W^{l+1T}}$ to get $\boldsymbol{\frac{∂C}{∂a_{j}^{l+1}}}$ (*Figure 17: blue elements*). Subsequently, the error is moved through the activation function by applying an elementwise multiplication by $\boldsymbol{g(z^{l+1}}$. The resulting vector $\boldsymbol{\frac{∂C}{∂z^{l}}}$ contains all errors for layer $l$  allowing to compute derivatives of $C$ regarding the parameters of layer $l$ with the equations above.

### Backpropagation in convolutional layers

![conv_comp_graph.png]({{site.baseurl}}/img/conv_comp_graph.png)
*Figure 18. Computational graph of a convolutional layer*

Just as fully-connected layers, convolutional layers can also be visualised in computational graph format. Thereby, two specific characteristics are revealed. At first the graph is sparse, meaning that the the neurons are not fully-connected. Secondly, weights are shared between neurons (*Fig 18*: colour-coded connections).

![conv_backprop_wei.png]({{site.baseurl}}/img/conv_backprop_wei.png)
*Figure 19. Computing *$$\frac{∂C}{∂w_{ij}^{l}}* in a convolutional layer*

To get the derivative of $C$ with respect to an arbitrary shared weight in a convolutional layer $$\frac{∂C}{∂w_{ij}^{l}}=a_{j}^{l-1}\frac{∂C}{∂z_{i}^{l}}$$ is computed for every pair of neurons connected by a shared weight. Subsequently the derivatives are summed. To compute $$\frac{∂C}{∂w_{ij}^{l}}$$ with regard to every weight of the convolutional layer, the steps can be represented as a convolution operation (Figure 19).

![conv_backprop_err.png]({{site.baseurl}}/img/conv_backprop_err2.png)
*Figure 20. Computing *$$\frac{∂C}{∂z_{i}^{l-1}}* in a convolutional layer*

Furthermore, moving backwards through the network, i.e. computing $$\frac{∂C}{∂z_{i}^{l-1}}$$, is also achieved by applying a convolution operation. This time a convolution operation utilising the inverted filters is applied to the errors of layer $l+1$. Thereby, the pattern described in Fig. 17 is reproduced. In other words. If an activation function comes into action during in a convolution layer, the errors are of course also moved backwards through them.

This concludes the my series about the inner workings of a convolutional neural network. In my opinion, these incredibly potent algorithms are far more intelligible than most people think and I hope after this intense reading I could show that in a convincing manner.



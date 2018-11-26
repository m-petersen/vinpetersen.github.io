---
layout: post
published: true
title: A Guided Tour Through a Convolutional Neural Network - Part 3
mathjax: true
math: true
subtitle: The Backward Pass
---
### The Cross-entropy Loss Function

The loss function quantifies the goodness of prediction by comparing predicted and true label. For a classification problem with a single true class, the cross-entropy loss function is a typical choice. 

$$C_{n}(a^{true},a^{pred})=-\sum_{class} a_{class}^{true}log(a_{class}^{pred})$$

Let the vector $\boldsymbol{a^{(2)}}$ which encompasses the prediction scores of the model be $$\boldsymbol{a^{pred}}$$.  $$\boldsymbol{a^{true}}$$ contains the true labels for every prediction, i.e. 1 for the true class, 0 otherwise. For every class, the true label is multiplied by the logarithm of the prediction. In fact, only the prediction score for the true class contributes to the overall loss result, since every other one is multiplied by zero. The product is negativised, resulting in a score that indicates a good agreement of predicted and true label through a low value. By this, an aim for the learning process is identified: lowering the loss function result. Since this whole procedure requires the predictions to be between 0 and 1, they are normalised beforehand.


### Backpropagation and Gradient Descent

![deriv.png]({{site.baseurl}}/img/deriv.png)

*Figure 11. Derivatives*

Like every other mathematical function, the cross-entropy loss function can be visualised in coordinate space (*Figure 11*). The cross-entropy result $$𝐶$$ is shown on the ordinate. It is a nonlinear function and as discussed in the last chapter its course depends on the model’s prediction value for the true class – $$C(\boldsymbol{a^{pred}})$$.

Obviously, a minimum is available, defining the value of $$𝐶$$ to reach via learning the model parameters. Backpropagation and gradient descent are the processes that achieve this descent resulting in an optimisation of the prediction results. How is explained using the fully-connected layer as an example in an iconic and simplified manner.

Before discussing both, some preliminaries considering the mathematical background are necessary. A derivative of the cross-entropy function $$𝐶$$ with respect to $$\boldsymbol{a^{pred}}$$ is denoted by $$\frac{dC}{\boldsymbol{da^{pred}}}$$ or alternatively $$C'(\boldsymbol{a^{pred}})$$ and describes to what degree $$𝐶$$ changes if $$\boldsymbol{a^{pred}}$$ is changed. Simply spoken, it is a function itself indicating the gradient of the function $$C(\boldsymbol{a^{pred}})$$ at any point $$\boldsymbol{a^{pred}}$$. It also displays how to change  $$\boldsymbol{a^{pred}}$$ to minimise $$C$$ – a fact that is exploited by backpropagation and gradient descent. Although easily achievable by a simply deriving $$𝐶$$, knowing $$\frac{dC}{\boldsymbol{da^{pred}}}$$ is not helpful for solving the optimisation problem of the network, since $$\boldsymbol{a^{pred}}$$ is no parameter. Contrary to this, weights and biases of the network are. Hence, knowing for example $$\frac{∂C}{∂w_{1,2}^{(2)}}$$ respectively knowing how to change $$w_{1,2}^{(2)}$$ to minimise $$𝐶$$ would be very helpful.

![backprop.png]({{site.baseurl}}/img/backprop.png)
*Figure 12. The purpose of backpropagation*

It is achieved by application of the chain rule, which will be subject of the next part. Now, it is not necessary for understanding the following paragraphs. Since the calculation of this specific derivative contains backward steps through the network parts connecting $$w_{1,2}^{(2)}$$ and $$𝐶$$, the process is called backpropagation (*Figure 12*). Having $$\frac{∂C}{∂w_{1,2}^{(2)}}$$ allows to draw the graph $$C(w_{1,2}^{(2)})$$. The reverse conclusion is, at every point of the graph a gradient is defined indicating the direction to move $$w_{1,2}^{(2)}$$ to minimise $$𝐶$$.

![grad_desc.png]({{site.baseurl}}/img/grad_desc.png)
*Figure 13. Gradient Descent*

Now, the optimisation is achieved via application of the gradient descent formula displayed in *Figure 13*: at first the gradient at the point $$w_{1,2}^{(2)}$$, the output of $$\frac{∂C}{∂w_{1,2}^{(2)}}$$, is multiplied with a learning rate variable. The product is subtracted from the initial value of $$w_{1,2}^{(2)}$$, resulting in an updated value for $$w_{1,2}^{(2)}$$ in the direction that minimises $$𝐶$$. This step is carried out until the algorithm reaches the function’s minimum and this point is called convergence. To be more detailed, the prorcess stops if $$C$$ and the $$C$$ of the preceding optimisation step do differ insignificantly. Whether they do is defined by a threshold defined beforehand.

Finally, the network parameters are optimised, which finalises the backward pass. This concatenation of steps is carried out for every learnable parameter of the network. Thereupon another cycle of forward and backward pass will be initialised, leading to a better prediction and further optimisation of weights. Typically, a lot more cycles are initialised to maximise the prediction quality.

Finally, it should be mentioned, that operating on non-training data is nothing more than exclusively performing the forward pass, i.e. no learning during the backward pass.

The final part of the series will take a deeper look into backpropagation in general and in the special case of convolutional layers.

### References and Inspiration
* [Deep Learning For Coders](https://course.fast.ai/) by Jeremy Howard and Fast.ai (v3 will be publicly available in early 2019)
* [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning) by Andrew Ng and deeplearning.ai

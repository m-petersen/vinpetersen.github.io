---
layout: post
published: false
title: A Guided Tour Through a Convolutional Neural Network - Part 3
---
### Backpropagation and Gradient Descent

<img src="https://github.com/vinpetersen/vinpetersen.github.io/tree/master/img/deriv.png" width="200" height="200" />

#![deriv.png]({{site.baseurl}}/img/deriv.png)
*Figure 6. Derivatives*

Like every other mathematical function, the cross-entropy loss function can be visualised in coordinate space (Figure 6). The cross-entropy result 𝐶 is shown on the ordinate. It is a nonlinear function and as discussed in the last chapter its course depends on the model’s prediction value for the true class – C(a^pred).

Obviously, a minimum is available, defining the value of 𝐶 to reach via learning the model parameters. Backpropagation and gradient descent are the processes that achieve this descent resulting in an optimisation of the prediction results. How is explained using the fully-connected layer as an example in an iconic and simplified manner.

Before discussing both, some preliminaries considering the mathematical background are necessary. A derivative of the cross-entropy function 𝐶 with respect to a^pred is denoted by dC/(da^pred ) or alternatively C'(a^pred) and describes to what degree 𝐶 changes if a^pred is changed. Simply spoken, it is a function itself indicating the gradient of the function at any point a^pred. It also displays how to change  a^pred to minimise C – a fact that is exploited by backpropagation and gradient descent. Although easily achievable by a simply deriving 𝐶, knowing dC/(da^pred ) is not helpful for solving the optimisation problem of the network, since a^pred is not changeable. Contrary to this, weights and biases of the network are. Hence, knowing for example ∂C/(〖∂w〗_1,2^((2)) )  respectively knowing how to change w_1,2^((2)) to minimise 𝐶 would be very helpful. 

![backprop.png]({{site.baseurl}}/img/backprop.png)
*Figure 7. Backpropagation*

It is achieved by application of the chain rule, which is beyond the scope of this work and not necessary for further understanding. Since the calculation of this specific derivative contains backward steps through the network parts connecting w_1,2^((2)) and 𝐶, it is called backpropagation (Figure 7). Having ∂C/(〖∂w〗_1,2^((2)) ) allows to draw the graph C(w_1,2^((2))). The reverse conclusion is, at every point of the graph a gradient is defined indicating the direction to move w_1,2^((2))  to minimise 𝐶.

![grad_desc.png]({{site.baseurl}}/img/grad_desc.png)
*Figure 8. Gradient Descent*

Now, the optimisation is achieved via application of the gradient descent formula displayed in Figure 8: at first the gradient at the point w_1,2^((2)) calculated by inputting it in ∂C/(〖∂w〗_1,2^((2)) )  is multiplied with a learning rate variable. The product is subtracted from the initial value of w_1,2^((2)), resulting in an updated value for w_1,2^((2)) in the direction that minimises 𝐶. This step is carried out until the algorithm reaches the function’s minimum and this point is called convergence. This practice is of course carried out for every learnable parameter in the network. 

Finally, the network parameters are optimised, which finalises the backward pass. This concatenation of steps is carried out for every parameter of the network. The optimisation process in convolutional layers differs slightly, but it will not be covered by this work since the core ideas are similar. 

Thereupon another cycle of forward and backward pass will be initialised, leading to a better prediction and further optimisation of weights. Typically, a lot more cycles are initialised to maximise the prediction quality.

Finally, it should be mentioned, that operating on non-training data is nothing more than exclusively performing the forward pass, i.e. no learning during the backward pass.

This concludes the series.

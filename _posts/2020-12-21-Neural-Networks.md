<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

## Neuron
<div align="center">
<img width="400" height="200" src="https://raw.githubusercontent.com/SharynHu/picBed/master/65E72571-A911-4B44-8ED5-CC45183AC035.png"/>
</div>

## Neural Network
Neural networks is combining multiple neurons together.
Below is the structure of a 2-layer neural network:

<div align="center">
<img width="400" height="200" src="https://raw.githubusercontent.com/SharynHu/picBed/master/B27A7ABB-D0F1-4953-93B2-8342635D7177.png"></img>
</div>

$a^{[0]}$ is the input layer, $a^{[1]}$ is the hidden layer, and $a^{[2]}$ is the output layer. For layer 1 the activation function is $g$ and for layer 2 the activation function is sigmoid. Parameter $w^{[1]}$ is a $4\times3$ matrix; parameter $b^{[1]}$ is a $4\times1$ vector.
For layer 1 we have

$$ W^{[1]}=\left[
\begin{matrix}
w^{[1]T}_1\\
w^{[1]T}_2\\
w^{[1]T}_3\\
w^{[1]T}_4\\
\end{matrix}
\right]$$

$$b^{[1]}=\left[
\begin{matrix}
b^{[1]}_1\\
b^{[1]}_2\\
b^{[1]}_3\\
b^{[1]}_4\\
\end{matrix}
\right]$$

$$z^{[1]}_1 = w^{[1]T}_1a^{[0]}+b^{[1]}_1, a^{[1]}_1=g(z^{[1]}_1)$$

$$z^{[1]}_2 = w^{[1]T}_2a^{[0]}+b^{[1]}_2, a^{[1]}_2=g(z^{[1]}_2)$$

$$z^{[1]}_3 = w^{[1]T}_3a^{[0]}+b^{[1]}_3, a^{[1]}_3=g(z^{[1]}_3)$$

$$z^{[1]}_4 = w^{[1]T}_4a^{[0]}+b^{[1]}_4, a^{[1]_4}=g(z^{[1]}_4)$$

Vectorize it we get:

$$z^{[1]}=W^{[1]}a^{[0]}, A^{[1]}=g(Z^{[1]})$$

Similarly for layer 2 we get

$$Z^{[2]}=W^{[2]}A^{[1]}, A^{[2]}=\sigma(z^{[2]})$$

## Vectorizing Across Multiple Examples
### Forward  Propagation
We define the training set $A^{[0]}$ to have m examples, that is

$$A^{[0]} = \left[
\begin{matrix}
a^{[0](1)]}, a^{[0](2)]}, \cdots, a^{[0](m)]}
\end{matrix}
\right]$$

Then we have 

$$Z^{[1]}=w^{[1]}A^{[0]}+b^{[1]}, A^{[1]}=g(Z^{[1]})$$

Similarly, 

$$Z^{[2]}=w^{[2]}A^{[1]}+b^{[2]}, A^{[2]}=\sigma(Z^{[2]})$$

### Cost Function
We define the cost function to be a cross entropy:

$$J(\hat Y, Y) = Y*\log\hat Y+(1-Y)*\log(1-\hat Y)$$

### Backward propagation
We know that if we use the sigmoid function for layer 2 and use the cross-entropy as the cost function, the gradient for $Z^{[2]}$ will be:

$$dZ^{[2]}=A^{[2]}-Y$$

$$dW^{[2]}=\frac{d\mathcal J}{dZ^{[2]}}$$

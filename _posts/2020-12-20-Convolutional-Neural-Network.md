## What is a CNN made of
A full convolutional neural network are made of:
1. input layer
2. **convolutional layer**
3. **ReLU layer**
4. **pooling layer**
5. fully-connected layer

## Why do we need CNN
1. We need to reserve the spacial information of input;
2. We need less parameters;
3. To avoid overfitting caused by large amount of parameters.

## 2-D convolution

<div align=center><img width="400" height="250" src="https://raw.githubusercontent.com/SharynHu/picBed/master/FBEB8B9C-513C-42DD-BA14-3ADC1E4C4144.gif"/></div>

### Commonly used techniques
#### Padding
<div align=center><img width="400" height="500" src="https://raw.githubusercontent.com/SharynHu/picBed/master/1_1okwhewf5KCtIPaFib4XaA.gif"/></div>

#### Stride

<div align=center><img width="400" height="400" src="https://raw.githubusercontent.com/SharynHu/picBed/master/57EEC4CF-CCAE-474B-8227-7E6AB3D0E7F2.gif"/></div>
<br>
**Note:** A stride of 1 is equivalent to a standard convolution.


## Multi-channal convolution
### Filter
A filter is actually a collection of kernals in general case.
One kernal for one channel.
For example:
for an image that has 3 channels, a filter has 3 kernals per channel.
<div align=center><img width="600" height="300" src="https://raw.githubusercontent.com/SharynHu/picBed/master/5D0EF5F7-4C9B-47B2-B142-268F211C69D6.gif"/></div>

Each of the per-channel processed versions are then summed together to form one channel. The kernels of a filter each produce one version of each channel, and the filter as a whole produces one overall output channel.

<div align=center><img width="700" height="300" src="https://raw.githubusercontent.com/SharynHu/picBed/master/074409D9-2155-4EE6-8A59-912895C8D5CC.gif"/></div>

Finally, then there’s the bias term. The way the bias term works here is that each output filter has one bias term. The bias gets added to the output channel so far to produce the final output channel.

<div align=center><img width="700" height="300" src="https://raw.githubusercontent.com/SharynHu/picBed/master/A46A9F9B-E39E-4827-9A86-ECED31387308.gif"/></div>

So for each filter, it produces one output channel. outputs for all filters concatenated are the final multi-channeled output.

## One by one convolution

# Reference
1. [https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
2. [https://zhuanlan.zhihu.com/p/47184529](https://zhuanlan.zhihu.com/p/47184529)
3. [https://iamaaditya.github.io/2016/03/one-by-one-convolution/](https://iamaaditya.github.io/2016/03/one-by-one-convolution/)

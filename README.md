# fp16-demo-tf

Some example codes for [mixed-precision training](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) in TensorFlow and PyTorch.

## General rule of thumb:
It's good to have parameters as multiple of 8 to utilize performance of [TensorCores](https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/) in [Volta GPUs](https://www.nvidia.com/en-us/data-center/tensorcore/).
- Convolutions: Multiple of 8 - Number of input channels, output channels, batch size
- GEMM: Multiple of 8 - M, N, K dimensions
- Fully connected layers: Multiple of 8 - Input features, output features, batch size

## The examples:
- [mnist_softmax.py](https://github.com/khcs/fp16-demo-tf/blob/master/mnist_softmax.py) - simple softmax mnist classification example in TensorFlow [source](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py)
- [mnist_softmax_fp16_naive.py](https://github.com/khcs/fp16-demo-tf/blob/master/mnist_softmax_fp16_naive.py) - naive fp16 implementation - just works
- [mnist_softmax_deep.py](https://github.com/khcs/fp16-demo-tf/blob/master/mnist_softmax_deep.py) - softmax mnist classification with 1 hidden layer
- [mnist_softmax_deep_fp16_naive.py](https://github.com/khcs/fp16-demo-tf/blob/master/mnist_softmax_deep_fp16_naive.py) - naive fp16 implementation of the [mnist_softmax_deep.py](https://github.com/khcs/fp16-demo-tf/blob/master/mnist_softmax_deep.py) - it doesn't work
- [mnist_softmax_deep_fp16_advanced.py](https://github.com/khcs/fp16-demo-tf/blob/master/mnist_softmax_deep_fp16_advanced.py) - mixed-precision implementation of the [mnist_softmax_deep.py](https://github.com/khcs/fp16-demo-tf/blob/master/mnist_softmax_deep.py) - works with speed-up utilizing TensorCores in Volta GPUs with reduced memory usage - can experiment with number of hidden units to see how that affects utilizing TensorCores and training speed
- [mnist_softmax_deep_conv_fp16_advanced.py](https://github.com/khcs/fp16-demo-tf/blob/master/mnist_softmax_deep_conv_fp16_advanced.py) - mixed-precision implementation of convolutional neural network for mnist classification - can experiment with convolutional filter size and if that affects utilizing TensorCores and training speed
- [pytorch](https://github.com/khcs/fp16-demo-tf/tree/master/pytorch) - corresponding PyTorch implementations

## Checking if TensorCores are utilized
- Run the program with nvprof and see the log output - if there's kernel calls with "884" then TensorCores are called.
Example:
```
nvprof python mnist_softmax_deep_conv_fp16_advanced.py
```
## Notes about loss-scaling
The "default" loss-scaling value of 128 works for all the examples here.
However, in a case it doesn't work, it's advised to choose a large value and gradually decrease it until sucessful.
[apex](https://github.com/NVIDIA/apex) is a easy-to-use mixed-precision training utilities for PyTorch, and it's [loss-scaler](https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/loss_scaler.py) does that.

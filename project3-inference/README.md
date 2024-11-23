# Project 3: Inference Optimization

This project builds upon a Lottery Ticket Hypothesis we performed for Machine Learning Theory.
Wights pruning is one of the techniques used to speed up inference. We will also explore other methods.

The Lottery Ticket Hypothesis introduced by J. Frankle and M. Carbin in 2018. We are using a convolutional neural network according to the specificaiton from the publication. It is a smaller variant of VGG.
The dataset used for this task is CIFAR 10.

Publication: https://arxiv.org/abs/1803.03635

### **The Lottery Ticket Hypothesis**

> A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations.

### Running the code

Open `project3-inference` in terminal and run:

```bash
python -m src.train_lth
```

### Architecture summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Conv6                                    [60, 10]                  --
├─Conv2d: 1-1                            [60, 64, 32, 32]          1,792
├─ReLU: 1-2                              [60, 64, 32, 32]          --
├─Conv2d: 1-3                            [60, 64, 32, 32]          36,928
├─ReLU: 1-4                              [60, 64, 32, 32]          --
├─MaxPool2d: 1-5                         [60, 64, 16, 16]          --
├─Conv2d: 1-6                            [60, 128, 16, 16]         73,856
├─ReLU: 1-7                              [60, 128, 16, 16]         --
├─Conv2d: 1-8                            [60, 128, 16, 16]         147,584
├─ReLU: 1-9                              [60, 128, 16, 16]         --
├─MaxPool2d: 1-10                        [60, 128, 8, 8]           --
├─Conv2d: 1-11                           [60, 256, 8, 8]           295,168
├─ReLU: 1-12                             [60, 256, 8, 8]           --
├─Conv2d: 1-13                           [60, 256, 8, 8]           590,080
├─ReLU: 1-14                             [60, 256, 8, 8]           --
├─MaxPool2d: 1-15                        [60, 256, 4, 4]           --
├─Flatten: 1-16                          [60, 4096]                --
├─Linear: 1-17                           [60, 256]                 1,048,832
├─ReLU: 1-18                             [60, 256]                 --
├─Linear: 1-19                           [60, 256]                 65,792
├─ReLU: 1-20                             [60, 256]                 --
├─Linear: 1-21                           [60, 10]                  2,570
==========================================================================================
Total params: 2,262,602
Trainable params: 2,262,602
Non-trainable params: 0
Total mult-adds (G): 9.25
==========================================================================================
Input size (MB): 0.74
Forward/backward pass size (MB): 110.35
Params size (MB): 9.05
Estimated Total Size (MB): 120.14
==========================================================================================
```

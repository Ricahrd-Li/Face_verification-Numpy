# A Hand-written LightCNN-9 implementation for face verification
---

Hi there! 

[](./figures/lightCNN9.png)

My roomate (Kelley Kuang) and I implemented a hand-written light-CNN-9 model (with 24 layers including 9 convolution layers)[[paper link]](https://arxiv.org/abs/1511.02683) for face verification on a subset LFW dataset. 

Here is a overview of the face verification system

[](./figure/system.png)

In training we train our DNN as a face classfication problem. Then in testing we use similarity score to verify if the input two images belong to the same person. 

We used numpy and img2col(Check the function `conv` in [`forward_layers.py`](./forward_layers.py) ) technique to acclerate convolution. 

For data preprosession, we use normalization and image cropping. 

After train for >30 epochs on training data, we get 0.70-0.75 F1 score on test data.  

---
The source file are organizd as follows:  

+ model architecture: [`main.py`](./main.py) 

+ Forward propagation(FP) computing: [`forward_layers.py`](./forward_layers.py) 

+ Backward propagation(BP) computing: [`backward_layers.py`](./backward_layers.py) 

+ Data reading and others: [`util.py`](./util.py) 

+ In addition, to find the best threadhold for face comparison (lead to best F1 score), we have [`compute_thr.m `](./compute_thr.m) to do grid search. 

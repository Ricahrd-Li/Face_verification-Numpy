# A Hand-written LightCNN-9 implementation for face verification
---

Hi there! 

My roomate (Kelley Kuang) and I implemented a hand-written light-CNN-9 model (with 24 layers including 9 convolution layers)[[paper link]](https://arxiv.org/abs/1511.02683) for face verification on a subset LFW dataset. 

We used numpy and img2col technique to acclerate convolution. 

After train for >30 epochs on training data， we get 0.70-0.75 F1 score on test data.  

---
The source file are organizd as follows:  

+ model architecture: [`main.py`](./main.py) 

+ Forward propagation(FP) computing: [`forward_layers.py`](./forward_layers.py) 

+ Backward propagation(BP) computing: [`backward_layers.py`](./backward_layers.py) 

+ Data reading and others: [`util.py`](./util.py) 

+ In addition, to find the best threadhold for face comparison (lead to best F1 score), we have [`compute_thr.m `](./compute_thr.m) to do grid search. 

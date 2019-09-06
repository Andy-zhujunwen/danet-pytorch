# danet-pytorch
[pytorch] DANet: Dual Attention Network for Scene Segmentation

# network：
![image](https://github.com/Andy-zhujunwen/danet-pytorch/blob/master/network.png)

# dataset:
cityspaces

# how to train:
write the dataset path in mypath.py and run 
```
python train.py
```
after trainning,it will save a model: danet.pth
## How to Inference：
#run:
```
python inference.py
```

# inference:
## input:
![image](https://github.com/Andy-zhujunwen/danet-pytorch/blob/master/danet/s1.jpeg)
## output:
![image](https://github.com/Andy-zhujunwen/danet-pytorch/blob/master/danet/testjpg.png)

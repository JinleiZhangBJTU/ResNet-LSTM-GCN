# Keras implementation of ResNet+LSTM+GCN (ResLSTM)
Keras implementation of ResNet
Keras implementation of Attention LSTM
Keras implementation of GCN

## [Deep-learning Architecture for Short-term Passenger Flow Forecasting in Urban Rail Transit](https://arxiv.org/abs/1912.12563)

<img src="https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN/blob/master/pictures/Model%20structure.png" width = "722" height = "813" alt="model structure" 
align=center>

## Description

We propose a deep-learning architecture combined residual network (ResNet), graph convolutional network (GCN) and long short-term memory (LSTM) (called “**ResLSTM**”) to forecast short-term passenger flow in urban rail transit on a network scale. First, improved methodologies of ResNet, GCN, and attention LSTM models are presented. Then, model architecture is proposed, wherein ResNet is used to capture deep abstract spatial correlations between subway stations, GCN is applied to extract network-topology information, and attention LSTM is used to extract temporal correlations. Model architecture includes four branches for inflow, outflow, graph-network topology, as well as weather conditions and air quality. To the best of our knowledge, this is the first time that air-quality indicators have been taken into account, and their influences on prediction precision have been quantified. Finally, ResLSTM is applied to Beijing subway. Three time granularities (10, 15, and 30 min) are chosen to conduct short-term passenger flow forecasting. Comparison of prediction performance of ResLSTM with those of many state-of-the-art models shows the advancement and robustness of ResLSTM. Moreover, comparison of prediction precisions obtained from time granularities of 10, 15, and 30 min indicates that prediction precision increases with increasing time granularity. 

## Data

The dimension of inflow data is n*time steps, where n represents number of stations and time steps denote time steps in 25 weekdays.

The structure of outflow data is the same with inflow data.

The dimension of meteorology data is n*time steps, where n represents  the 11 meteorology indicators such as temperature, wind speed.

## Requirement

Keras == 2.2.4 

tensorflow-gpu == 1.10.0

numpy == 1.14.5

scipy == 1.3.3

scikit-learn == 0.20.2

protobuf == 3.6.0

## Implementation

Just download this repository and using PyCharm to open it. Then run ResLSTM.py.

## Result

![Model comparison](https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN/blob/master/pictures/Model%20comparison.jpg)

## Reference

Zhang, Jinlei, Feng Chen, Yadi Zhu, and Yinan Guo. "[Deep-learning Architecture for Short-term Passenger Flow Forecasting in Urban Rail Transit.](https://arxiv.org/abs/1912.12563)" arXiv preprint arXiv:1912.12563 (2019).

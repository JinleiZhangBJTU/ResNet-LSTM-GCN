# _*_coding:utf-8_*_
import csv
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt

os.chdir('D:/论文2/ResLSTM/10/')

data_predict=[]
with open('160-predictions.csv',"r") as file:
	#data = csv.reader(file, delimiter=",")
	for line in file:
		line=line.replace('"','').strip().split(',')
		line=[float(x) for x in line]
		data_predict.append(line)
#删掉前10行
data_predict=data_predict[10:len(data_predict)][:]
print(len(data_predict))
data_old_merge=[[] for i in range(int(len(data_predict)/3))]
for i in range(int(len(data_predict)/3)):
	for j in range(len(data_predict[0])):
		data_old_merge[i].append(data_predict[i*3][j]+data_predict[i*3+1][j]+data_predict[i*3+2][j])
print(len(data_old_merge))

data_original=[]
with open('160-Y_test_original.csv',"r") as file:
	#data = csv.reader(file, delimiter=",")
	for line in file:
		line=line.replace('"','').strip().split(',')
		line=[float(x) for x in line]
		data_original.append(line)
#删掉前10行
data_original=data_original[10:len(data_original)][:]
print(len(data_original))
data_original_merge=[[] for i in range(int(len(data_original)/3))]
for i in range(int(len(data_original)/3)):
	for j in range(len(data_original[0])):
		data_original_merge[i].append(data_original[i*3][j]+data_original[i*3+1][j]+data_original[i*3+2][j])
print(len(data_original_merge))


#定义平均绝对百分比误差和评价函数
def weighted_mean_absolute_percentage_error(Y_true, Y_pred):
	#两个矩阵都是n行276列
	total_sum=np.sum(Y_true)
	average=[]
	for i in range(len(Y_true)):
		for j in range(len(Y_true[0])):
			if Y_true[i][j]>0:
				#加权   (y_true[i][j]/np.sum(y_true[i]))*
				temp=(Y_true[i][j]/total_sum)*np.abs((Y_true[i][j] - Y_pred[i][j]) / Y_true[i][j])
				average.append(temp)
	return np.sum(average)

def evaluate_performance(Y_test_original,predictions):
	RMSE = sqrt(mean_squared_error(Y_test_original, predictions))
	print('均方根误差RMSE是'+str(RMSE))
	R2 = r2_score(Y_test_original,predictions)
	print("R2是："+str(R2))
	MAE=mean_absolute_error(Y_test_original, predictions)
	print("平均绝对误差MAE是："+str(MAE))
	WMAPE=weighted_mean_absolute_percentage_error(Y_test_original,predictions)
	print("WMAPE是"+str(WMAPE))
	return RMSE,R2,MAE,WMAPE

RMSE,R2,MAE,WMAPE=evaluate_performance(data_original_merge,data_old_merge)


def Save_Data(Y_test_original,predictions,RMSE,R2,MAE,WMAPE):
	RMSE_merge=[]
	R2_merge=[]
	MAE_merge=[]
	WMAPE_merge=[]
	RMSE_merge.append(RMSE)
	R2_merge.append(R2)
	MAE_merge.append(MAE)
	WMAPE_merge.append(WMAPE)
	np.savetxt('RMSE_merge.txt',RMSE_merge)
	np.savetxt('R2_merge.txt',R2_merge)
	np.savetxt('MAE_merge.txt',MAE_merge)
	np.savetxt('WMAPE_merge.txt',WMAPE_merge)
	with open('predictions.csv','w') as file:
		for i in range(len(predictions)):
			file.write(str(predictions[i]).replace("'","").replace("[","").replace("]","")+"\n")
	with open('Y_test_original.csv','w') as file:
		for i in range(len(Y_test_original)):
			file.write(str(Y_test_original[i]).replace("'","").replace("[","").replace("]","")+"\n")


Save_Data(data_original_merge,data_old_merge,RMSE,R2,MAE,WMAPE)


# #写入矩阵
# print(len(Weather_data_new))
# with open('E:/0博士培养/第一部分客流相关性分析及车站聚类/论文2/ResNet预测/天气数据/Weather数据60分钟时间粒度-归一化.csv',"w") as file:
# 	for i in range(len(Weather_data_new)):
# 		file.write(str(Weather_data_new[i]).replace("[",'').replace("]",'').replace("'",'')+'\n')

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
np.set_printoptions(threshold=np.inf)
import time, os
import keras
keras.backend.set_image_data_format('channels_last')
from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model #visualize model
from keras.models import load_model
from keras.optimizers import Adam
from metrics import evaluate_performance
from load_data import Get_All_Data

# os.chdir('D:/论文2/upload to GitHub/')
os.environ["PATH"] += os.pathsep + 'E:/Program Files (x86)/Graphviz2.38/bin' #used for visualizing the model

global_start_time = time.time()

def Unit(x, filters, pool=False):
	res = x
	if pool:
		x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
		res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)
	out = BatchNormalization()(x)
	out = Activation("relu")(out)
	out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

	out = BatchNormalization()(out)
	out = Activation("relu")(out)
	out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

	out = keras.layers.add([res, out])

	return out

def attention_3d_block(inputs,timesteps):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(timesteps, activation='linear')(a)
    a_probs = Permute((2, 1))(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

# Define the model
def multi_input_model(time_lag):
    """build multi input model构建多输入模型"""
    input1_ = Input(shape=(276, time_lag-1, 3), name='input1')
    input2_ = Input(shape=(276, time_lag-1, 3), name='input2')
    input3_ = Input(shape=(276, time_lag-1, 1), name='input3')
    input4_ = Input(shape=(11, time_lag-1, 1), name='input4')
    #first input
    x1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input1_)
    x1 = Unit(x1, 32)
    x1 = Unit(x1, 64, pool=True)
    x1 = Flatten()(x1)
    x1 = Dense(276)(x1)

    # second input
    x2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input2_)
    x2 = Unit(x2, 32)
    x2 = Unit(x2, 64, pool=True)
    x2 = Flatten()(x2)
    x2 = Dense(276)(x2)

    # third input
    x3 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input3_)
    x3 = Unit(x3, 32)
    x3 = Unit(x3, 64, pool=True)
    x3 = Flatten()(x3)
    x3 = Dense(276)(x3)

    # fourth input
    x4 = Flatten()(input4_)
    x4 = Dense(276)(x4)
    x4 = Reshape(target_shape=(276, 1))(x4)
    x4 = LSTM(128, return_sequences=True, input_shape=(276, 1))(x4)
    x4 = LSTM(276, return_sequences=False)(x4)
    x4 = Dense(276)(x4)

    out = keras.layers.add([x1, x2, x3, x4])
    out = Reshape(target_shape=(276, 1))(out)
    out = LSTM(128, return_sequences=True,input_shape=(276, 1))(out)
    out = attention_3d_block(out, 276)#shape of the output is（276，128）
    out = Flatten()(out)
    out = Dense(276)(out)

    model = Model(inputs=[input1_, input2_, input3_,input4_], outputs=[out]) #[input1_, input2_, input3_]
    return model

def build_model(X_train_1,X_train_2,X_train_3,X_train_4,Y_train,X_test_1,X_test_2,X_test_3,X_test_4,Y_test,\
	Y_test_original,batch_size,epochs,a,time_lag):

	X_train_1 = X_train_1.reshape(X_train_1.shape[0],  276, time_lag-1, 3)
	X_train_2 = X_train_2.reshape(X_train_2.shape[0],  276, time_lag-1, 3)
	X_train_3 = X_train_3.reshape(X_train_3.shape[0],  276, time_lag-1, 1)
	X_train_4 = X_train_4.reshape(X_train_4.shape[0],  11, time_lag-1, 1)
	Y_train = Y_train.reshape(Y_train.shape[0], 276)

	X_test_1 = X_test_1.reshape(X_test_1.shape[0],  276, time_lag-1, 3)
	X_test_2 = X_test_2.reshape(X_test_2.shape[0],  276, time_lag-1, 3)
	X_test_3 = X_test_3.reshape(X_test_3.shape[0],  276, time_lag-1, 1)
	X_test_4 = X_test_4.reshape(X_test_4.shape[0],  11, time_lag-1, 1)
	Y_test = Y_test.reshape(Y_test.shape[0], 276)

	if epochs == 50:
		model = multi_input_model(time_lag)
		model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
		model.fit([X_train_1, X_train_2, X_train_3, X_train_4], Y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=False)#, validation_split=0.05
		output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size=batch_size)
	else:
		# train models every 10 epoches
		model = load_model('testresult/'+str(epochs-10)+'-model-with-graph.h5')
		model.fit([X_train_1, X_train_2, X_train_3, X_train_4], Y_train, batch_size=batch_size, epochs=10, verbose=2, shuffle=False)# , validation_split=0.05
		output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size=batch_size)

	#rescale the output of this model将输出进行反归一化
	predictions = np.zeros((output.shape[0], output.shape[1]))
	for i in range(len(predictions)):
		for j in range(len(predictions[0])):
			predictions[i, j] = round(output[i, j]*a, 0)
			if predictions[i, j] < 0:
				predictions[i, j] = 0

	RMSE,R2,MAE,WMAPE=evaluate_performance(Y_test_original,predictions)
	#visualize the model structure
	plot_model(model, to_file='model.png', show_shapes=True)
	#print(model.summary())

	return model,Y_test_original,predictions,RMSE,R2,MAE,WMAPE

def Save_Data(path,model,Y_test_original,predictions,RMSE,R2,MAE,WMAPE,Run_epoch):
	print(Run_epoch)
	RMSE_ALL=[]
	R2_ALL=[]
	MAE_ALL=[]
	WMAPE_ALL=[]
	Average_train_time=[]
	RMSE_ALL.append(RMSE)
	R2_ALL.append(R2)
	MAE_ALL.append(MAE)
	WMAPE_ALL.append(WMAPE)
	model.save(path+str(Run_epoch)+'-model-with-graph.h5')
	np.savetxt(path+str(Run_epoch)+'-RMSE_ALL.txt', RMSE_ALL)
	np.savetxt(path+str(Run_epoch)+'-R2_ALL.txt', R2_ALL)
	np.savetxt(path+str(Run_epoch)+'-MAE_ALL.txt', MAE_ALL)
	np.savetxt(path+str(Run_epoch)+'-WMAPE_ALL.txt', WMAPE_ALL)
	with open(path+str(Run_epoch)+'-predictions.csv', 'w') as file:
		predictions = predictions.tolist()
		for i in range(len(predictions)):
			file.write(str(predictions[i]).replace("'", "").replace("[", "").replace("]", "")+"\n")
	with open(path+str(Run_epoch)+'-Y_test_original.csv', 'w') as file:
		Y_test_original = Y_test_original.tolist()
		for i in range(len(Y_test_original)):
			file.write(str(Y_test_original[i]).replace("'", "").replace("[", "").replace("]", "")+"\n")
	duration_time = time.time() - global_start_time
	Average_train_time.append(duration_time)
	np.savetxt(path+str(Run_epoch)+'-Average_train_time.txt', Average_train_time)
	print('total training time(s):', duration_time)

X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b,X_train_2,X_test_2,X_train_3,X_test_3,X_train_4,X_test_4=\
	Get_All_Data(TG=15, time_lag=6, TG_in_one_day=72, forecast_day_number=5, TG_in_one_week=360)
Run_epoch = 50  # first training 50 epoch, and then add 10 epoch every time 初始训练epoch，以后每次加10，运行15次
for i in range(15):
	model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE = build_model(X_train_1,X_train_2,X_train_3,X_train_4,Y_train,X_test_1,X_test_2,X_test_3,X_test_4,Y_test,\
		Y_test_original,batch_size=64,epochs=Run_epoch,a=a,time_lag=6)
	Save_Data("testresult/", model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, Run_epoch)
	Run_epoch += 10


#For Get_All_Data, change parameters referring to this: TG=15, time_lag=6, TG_in_one_day=72, forecast_day_number=5, TG_in_one_week=360
#10min:10,6,108,5,540,eopch=200
#15min:15,6,72,5,360 eopch=140
#30min:30,6,36,5,180 eopch=200
#60min:60,6,18,5,90 eopch=235
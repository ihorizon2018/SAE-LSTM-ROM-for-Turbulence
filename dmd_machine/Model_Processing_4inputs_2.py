import os
import vtktools
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from cycler  import cycler
from sklearn.metrics import mean_squared_error
from keras import backend as K
#from dmd_machine import vtkOperatorimport vtk
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from matplotlib.ticker import FormatStrFormatter

def root_mean_squared_error(true, pred):
    return K.sqrt(K.mean(K.square(pred - true))) 


def save_model(model, model_name, save_dir):
# function for saving model
	
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)

def draw_plot(data):
    plt.figure(1)
    plt.plot(data)
    plt.xlabel('n',{'size' : 11})
    plt.ylabel('value',{'size' : 11})
    plt.show()



def draw_Acc_Loss(history):
# draw the plot for loss and acc
	plt.figure(1)
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

def pearson_value(ori_data, rom_data):
    # print(ori_data.shape)
    # print(len(ori_data))
    pearson_value = []
    if len(ori_data) != len(rom_data):
        print('the length of these two array do not match')
    else:
        for i in range(len(rom_data)):
            row_1 = np.reshape(ori_data[i],(-1,1))
            row_2 = np.reshape(rom_data[i],(-1,1))
            data = np.hstack((row_1,row_2))
            data = pd.DataFrame(data)
            pearson = data.corr() # pearson cc  # 
            # pearson = data.corr('spearman') # spearman cc  
            pear_value=pearson.iloc[0:1,1:2]
            value = pear_value.values
            if i == 0:
                pearson_value = value
            else:
                pearson_value = np.hstack((pearson_value,value)) 
        pearson_value = np.reshape(pearson_value,(-1,1))
    return np.array(pearson_value)

def pcc_of_two(ori_data, rom_data):

    if ori_data.ndim == rom_data.ndim:
        if rom_data.ndim == 3:
            y_u = ori_data[...,0] # u
            y_v = ori_data[...,1] # v
            y_0_u = rom_data[...,0] # u
            y_0_v = rom_data[...,1] # v
            pcc_x = pearson_value(y_u, y_0_u)
            pcc_y = pearson_value(y_v, y_0_v)
            plt.figure(1)
            plt.plot(pcc_x)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('Pearson Correlation Coefficient of x axis',{'size' : 11})
            plt.figure(2)
            plt.plot(pcc_y)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('Pearson Correlation Coefficient of y axis',{'size' : 11})
            plt.show()
        elif rom_data.ndim == 2:
            pcc = pearson_value(ori_data, rom_data)
            # print(pcc.shape)
            plt.figure(1)
            # x = np.linspace(0,pcc.shape[0], num = pcc.shape[0])
            plt.plot(pcc)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('Pearson Correlation Coefficient',{'size' : 11})
            plt.show()
    else:
        print('the dimension of these two series are not equal. Please check them.')


def cc(ori_data, rom_data_0, rom_data_1,rom_data_2,y_axis_min,fieldName,startNumber,loc,n):
    # , *rom_data_4, *rom_data_5):
# draw the plot for correlation coefficient

    if ori_data.ndim == rom_data_0.ndim:
        if rom_data_0.ndim == 3:
            
            y_u = ori_data[...,0] # u
            y_v = ori_data[...,1] # v
            y_0_u = rom_data_0[...,0] # u
            y_0_v = rom_data_0[...,1] # v
            y_1_u = rom_data_1[...,0] # u
            y_1_v = rom_data_1[...,1] # v


            pcc_x = pearson_value(y_u, y_0_u)
            pcc_y = pearson_value(y_v, y_0_v)
            pcc_1_x = pearson_value(y_u, y_1_u)
            pcc_1_y = pearson_value(y_v, y_1_v)


            x = np.linspace(0,pcc_x.shape[0]/10,pcc_x.shape[0])

            plt.figure(1)
            plt.plot(x, pcc_x,'--', x, pcc_1_x, '--', linewidth = 0.7)
            plt.ylim((0.9, 1.0001))
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('Pearson Correlation Coefficient of x axis',{'size' : 11})
            plt.legend(['PCA+TF', 'AE+TF dim_2', 'AE+TF dim_6', 'AE+TF dim_8'], loc='lower right') 
            plt.figure(2)
            plt.plot(x, pcc_y, '--',x, pcc_1_y,'--', '--', linewidth = 0.7)
            plt.ylim((0.85, 1.0001))
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('Pearson Correlation Coefficient of y axis',{'size' : 11})
            plt.legend(['PCA+TF', 'AE+TF dim_2', 'AE+TF dim_6', 'AE+TF dim_8'], loc='lower right')  

            plt.show()


        elif rom_data_0.ndim == 2:    
            pcc_0 = pearson_value(ori_data, rom_data_0)
            pcc_1 = pearson_value(ori_data, rom_data_1)
            pcc_2 = pearson_value(ori_data, rom_data_2)
            #pcc_3 = pearson_value(ori_data, rom_data_3)
            # pcc_4 = pearson_value(ori_data, rom_data_4)
            # pcc_5 = pearson_value(ori_data, rom_data_5)
            # time = [0,15]# [start point, finish point]->200s

            fig=plt.figure(figsize=(10, 7))
            fig.dpi=200
            ax0 = fig.add_subplot()
            ax0.set_prop_cycle(color = ['#f6b93b','#BF3EFF','#104E8B','#8DEEEE'], linestyle = ['-', '-', '-', '-'])
            plt.ylim((0, 1.001))

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.xlabel('Time(s)',{'size' : 15})
            plt.ylabel('Pearson Correlation Coefficient',{'size' : 15})
            
                      
            # y_0 = pcc_0[-1800:-1200,:]
            # y_1 = pcc_1[-1800:-1200,:]
            # y_2 = pcc_2[-1800:-1200,:]
            # y_3 = pcc_3[-1800:-1200,:]
            # y_4 = pcc_4[-1800:-1200,:]
            # y_5 = pcc_5[-1800:-1200,:]
            # plt.title('Correlation Coefficient')
          
            #print(pcc_0.shape, pcc_1.shape)
            # plt.plot(x, pcc_0, x, pcc_1, x, pcc_2, x, pcc_3, x, y_4, x, y_5,linewidth = 0.8)
   
             
            if fieldName == 'Velocity':
                number=14
                rate=10
                x = np.linspace(startNumber+0, startNumber+(pcc_0.shape[0]-number) / rate, pcc_0.shape[0]-number)[1:]
                y_0=pcc_0
                y_1 = pcc_1
                y_2 = pcc_2

                y_0 = moving_average(pcc_0, number)[int(number/2)+1:pcc_0.shape[0]-int(number/2)]
                y_1 = moving_average(pcc_1, number)[int(number/2)+1:pcc_0.shape[0]-int(number/2)]
                y_2 = moving_average(pcc_2, number)[int(number/2)+1:pcc_0.shape[0]-int(number/2)]
                #y_2 = moving_average(pcc_2, number)
                #y_3 = moving_average(pcc_3, number)

                #y_2 = pcc_2
                #y_3 = pcc_3
                plt.ylim((y_axis_min, 1.001))
                ax0.plot(x, y_0, x, y_1,x, y_2,linewidth=1.5)
                plt.legend(['SAE-LSTM-bunch', 'SAE-LSTM-individual','SAE-DMD'], loc=loc,fontsize=15,ncol=n)

            
            elif fieldName == 'U':
                point = [0,ori_data.shape[0]]
                #point = [2800,3250]
                #rate=0.1
                rate=10
                number=14
                y_0 = pcc_0
                y_1 = pcc_1
                y_2 = pcc_2
                #y_3 = pcc_3
                y_0 = moving_average(pcc_0, number)[int(number/2)+1:pcc_0.shape[0]-int(number/2)]
                y_1 = moving_average(pcc_1, number)[int(number/2)+1:pcc_0.shape[0]-int(number/2)]
                y_2 = moving_average(pcc_2, number)[int(number/2)+1:pcc_0.shape[0]-int(number/2)]
                #y_2 = moving_average(pcc_2, number)
                #y_3 = moving_average(pcc_3, number)
                x = np.linspace(startNumber+0, startNumber+(pcc_0.shape[0]-number) / rate, pcc_0.shape[0]-number)[1:]
                #ax.plot(x, y_0[point[0]:point[1],:], x, y_1[point[0]:point[1],:], x, y_2[point[0]:point[1],:], x, y_3[point[0]:point[1],:])
                plt.ylim((y_axis_min, 1.006))
                ax0.plot(x, y_0, x, y_1,x, y_2,linewidth=1.5)
                
                plt.xlabel('Time(s)',{'size' : 15})

                plt.legend(['SAE-LSTM-bunch', 'SAE-LSTM-individual','SAE-DMD'], loc=loc,fontsize=15,ncol=n)
            plt.show()
    else:
        print('the dimension of these two series are not equal. Please check them.')

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)

    return np.convolve(interval[:,0], window, 'same')

# draw the plot for point over time series
def point_over_time(ori_data, rom_data_0, rom_data_1, rom_data_2,pointNo, fieldName,startNumber,led_location,n):
 # rom_data_2, rom_data_4, rom_data_5, pointNo, fieldName):# 1: full model; 2: ROM model
    
    if ori_data.shape != rom_data_0.shape:
        print('the shape of these two series do not match. Please check them.')
        return 


    if fieldName == 'Velocity': # flow_past_cylinder
        point = [0,ori_data.shape[0]] # [start point, finish point]->200s
        #point = [0,2000]
        rate = 10
        # ylim_x = [0.45, 0.56] # ylim for first figure->data in x axis
        # ylim_y = [0, 0.1] # ylim for second figure->data in y axis
            
        
        # time = int((point[1]-point[0])/rate)
        # x = np.linspace(point[0],point[1],time*rate) 
        x = np.linspace(startNumber+point[0]/rate,startNumber+point[1]/rate,int(point[1]-point[0]))
        # y_u = ori_data[point[0]:point[1],pointNo,0] # u
        # y_v = ori_data[point[0]:point[1],pointNo,1] # v
        # y_0_u = rom_data_0[point[0]:point[1],pointNo,0] # u
        # y_0_v = rom_data_0[point[0]:point[1],pointNo,1] # v
        # y_1_u = rom_data_1[point[0]:point[1],pointNo,0] # u
        # y_1_v = rom_data_1[point[0]:point[1],pointNo,1] # v
        # y_2_u = rom_data_2[point[0]:point[1],pointNo,0] # u
        # y_2_v = rom_data_2[point[0]:point[1],pointNo,1] # v
       
        y = ori_data[point[0]:point[1],pointNo]
        y_0 = rom_data_0[point[0]:point[1],pointNo]
        y_1 = rom_data_1[point[0]:point[1],pointNo]
        y_2 = rom_data_2[point[0]:point[1],pointNo]
        #y_3 = rom_data_3[point[0]:point[1],pointNo]
        #

        #
        fig=plt.figure(figsize=(18, 6))
        fig.dpi=200
        ax0 = fig.add_subplot()
        #fig, ax0 = plt.subplots()
        ax0.set_prop_cycle(color = ['#000000','#f6b93b','#FF3030','#104E8B','#8DEEEE'], linestyle = ['-','-', '-', '-', '-'])#linestyle = ['-',':', '--', ':', '-']
        # ax0.plot(x, y, x, y_0, x, y_1, x, y_2, x, y_3, linewidth = 0.8)
        ax0.plot(x, y, x, y_0, x, y_1, x, y_2, linewidth = 1.5)
        
        # plt.figure(1)
        # plt.plot(x, y_u,'k', x, y_0_u, 'r--', x, y_1_u, 'g--', linewidth = 0.7)
        # plt.xlim((point[0]/rate, point[1]/rate))# range
        # plt.ylim((ylim_x[0], ylim_x[1]))
        #ax0.set_xlim([0, 200])
        #ax0.set_ylim([0.30, 0.85])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        #plt.title(fieldName + ' Magnitude')
        plt.xlabel('Time(s)',fontsize=15)
        plt.ylabel(fieldName + ' Magnitude',fontsize=15)
        plt.legend(['Full Model', 'SAE-LSTM-bunch m=5', 'SAE-LSTM-bunch m=10', 'SAE-LSTM-bunch m=45'], fontsize=15,loc=led_location,frameon=True,ncol=n)

        plt.show()

    elif fieldName == 'U': # two bridge cylinder
        point = [0,rom_data_0.shape[0]]
        #point = [0,2000]
        #rate = 10
        rate=10
        # ylim_x = [0.45, 0.56] # ylim for first figure->data in x axis
        # ylim_y = [0, 0.1] # ylim for second figure->data in y axis
            
        
        # time = int((point[1]-point[0])/rate)
        # x = np.linspace(point[0],point[1],time*rate) 
        x = np.linspace(startNumber+0,startNumber+point[1]/rate,int(point[1]-point[0]))
        # y_u = ori_data[point[0]:point[1],pointNo,0] # u
        # y_v = ori_data[point[0]:point[1],pointNo,1] # v
        # y_0_u = rom_data_0[point[0]:point[1],pointNo,0] # u
        # y_0_v = rom_data_0[point[0]:point[1],pointNo,1] # v
        # y_1_u = rom_data_1[point[0]:point[1],pointNo,0] # u
        # y_1_v = rom_data_1[point[0]:point[1],pointNo,1] # v
        # y_2_u = rom_data_2[point[0]:point[1],pointNo,0] # u
        # y_2_v = rom_data_2[point[0]:point[1],pointNo,1] # v
       
        y = ori_data[point[0]:point[1],pointNo]
        y_0 = rom_data_0[point[0]:point[1],pointNo]
        y_1 = rom_data_1[point[0]:point[1],pointNo]
        y_2 = rom_data_2[point[0]:point[1],pointNo]
        #y_3 = rom_data_3[point[0]:point[1],pointNo]
        #
        
        #
        fig=plt.figure(figsize=(18, 6))
        ax0 = fig.add_subplot()
        #fig, ax0 = plt.subplots()
        ax0.set_prop_cycle(color = ['#000000','#f6b93b','#FF3030','#104E8B','#8DEEEE'], linestyle = ['-','-', '-', '-', '-'])
        # ax0.plot(x, y, x, y_0, x, y_1, x, y_2, x, y_3, linewidth = 0.8)
        ax0.plot(x, y, x, y_0, x, y_1, x, y_2, linewidth = 1.5)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #ax0.set_ylim([0.40, 6.0])
        
        #plt.title('Velocity' + ' Magnitude')
        plt.xlabel('Time(s)',{'size' : 15})
        plt.ylabel('Velocity Magnitude',{'size' : 15})
       
        plt.legend(['Full Model', 'SAE-LSTM-bunch m=7', 'SAE-LSTM-bunch m=12', 'SAE-LSTM-bunch m=35'], loc=led_location,fontsize=15,ncol=n)


        plt.show()

    else:
        print(' Please check the field name.')

def point_over_time_error(ori_data, rom_data_0, rom_data_1, rom_data_2, pointNo, fieldName, startNumber,
                    led_location,n):
    # rom_data_2, rom_data_4, rom_data_5, pointNo, fieldName):# 1: full model; 2: ROM model

    if ori_data.shape != rom_data_0.shape:
        print('the shape of these two series do not match. Please check them.')
        return
        
    
    if fieldName == 'Velocity':  # flow_past_cylinder
        point = [0, ori_data.shape[0]]  # [start point, finish point]->200s
        # point = [0,2000]
        rate = 10
        # ylim_x = [0.45, 0.56] # ylim for first figure->data in x axis
        # ylim_y = [0, 0.1] # ylim for second figure->data in y axis

        # time = int((point[1]-point[0])/rate)
        # x = np.linspace(point[0],point[1],time*rate)
        x = np.linspace(startNumber + point[0] / rate, startNumber + point[1] / rate, int(point[1] - point[0]))
        # y_u = ori_data[point[0]:point[1],pointNo,0] # u
        # y_v = ori_data[point[0]:point[1],pointNo,1] # v
        # y_0_u = rom_data_0[point[0]:point[1],pointNo,0] # u
        # y_0_v = rom_data_0[point[0]:point[1],pointNo,1] # v
        # y_1_u = rom_data_1[point[0]:point[1],pointNo,0] # u
        # y_1_v = rom_data_1[point[0]:point[1],pointNo,1] # v
        # y_2_u = rom_data_2[point[0]:point[1],pointNo,0] # u
        # y_2_v = rom_data_2[point[0]:point[1],pointNo,1] # v

        y = ori_data[point[0]:point[1], pointNo]
        y_0 = rom_data_0[point[0]:point[1], pointNo]
        y_1 = rom_data_1[point[0]:point[1], pointNo]
        y_2 = rom_data_2[point[0]:point[1], pointNo]
        #y_3 = rom_data_3[point[0]:point[1], pointNo]
        #

        y_0 = y-y_0
        y_1 = y-y_1
        y_2 = y-y_2
        #y_3 = y-y_3
        #
        fig = plt.figure(figsize=(18, 6))
        fig.dpi = 200
        ax0 = fig.add_subplot()
        # fig, ax0 = plt.subplots()
        ax0.set_prop_cycle(color=['#f6b93b','#FF3030','#104E8B','#8DEEEE'],linestyle=['-', '-', '-', '-'])
                           #linestyle=[':', '--', ':', '-'])
        # ax0.plot(x, y, x, y_0, x, y_1, x, y_2, x, y_3, linewidth = 0.8)
        ax0.plot(x, y_0, x, y_1, x, y_2, linewidth=1.5)

        # plt.figure(1)
        # plt.plot(x, y_u,'k', x, y_0_u, 'r--', x, y_1_u, 'g--', linewidth = 0.7)
        # plt.xlim((point[0]/rate, point[1]/rate))# range
        # plt.ylim((ylim_x[0], ylim_x[1]))
        # ax0.set_xlim([0, 200])
        # ax0.set_ylim([0.30, 0.85])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.ylim((-0.06, 0.06))

        # plt.title(fieldName + ' Magnitude')
        plt.xlabel('Time(s)', fontsize=15)
        plt.ylabel(fieldName + ' Magnitude Error', fontsize=15)
        # plt.legend(['Full Model', 'ROM(AE+TF)', 'ROM(PCA+TF)'], loc='lower right')
        plt.legend(['SAE-LSTM-bunch m=5', 'SAE-LSTM-bunch m=10', 'SAE-LSTM-bunch m=45'], fontsize=15,
                   loc=led_location,ncol=n)

        plt.show()

    elif fieldName == 'U':  # two bridge cylinder
        point = [0, rom_data_0.shape[0]]
        # point = [0,2000]
        rate = 10
        #rate = 0.01
        # ylim_x = [0.45, 0.56] # ylim for first figure->data in x axis
        # ylim_y = [0, 0.1] # ylim for second figure->data in y axis

        # time = int((point[1]-point[0])/rate)
        # x = np.linspace(point[0],point[1],time*rate)
        x = np.linspace(startNumber + 0, startNumber + point[1] / rate, int(point[1] - point[0]))
        # y_u = ori_data[point[0]:point[1],pointNo,0] # u
        # y_v = ori_data[point[0]:point[1],pointNo,1] # v
        # y_0_u = rom_data_0[point[0]:point[1],pointNo,0] # u
        # y_0_v = rom_data_0[point[0]:point[1],pointNo,1] # v
        # y_1_u = rom_data_1[point[0]:point[1],pointNo,0] # u
        # y_1_v = rom_data_1[point[0]:point[1],pointNo,1] # v
        # y_2_u = rom_data_2[point[0]:point[1],pointNo,0] # u
        # y_2_v = rom_data_2[point[0]:point[1],pointNo,1] # v

        y = ori_data[point[0]:point[1], pointNo]
        y_0 = rom_data_0[point[0]:point[1], pointNo]
        y_1 = rom_data_1[point[0]:point[1], pointNo]
        y_2 = rom_data_2[point[0]:point[1], pointNo]
        #y_3 = rom_data_3[point[0]:point[1], pointNo]
        #
        y_0 = y-y_0
        y_1 = y-y_1
        y_2 = y-y_2
        #
        fig = plt.figure(figsize=(18, 6))
        ax0 = fig.add_subplot()
        # fig, ax0 = plt.subplots()
        ax0.set_prop_cycle(color=['#f6b93b', '#FF3030', '#104E8B', '#8DEEEE'],
                           linestyle=['-', '-', '-', '-'])
        # ax0.plot(x, y, x, y_0, x, y_1, x, y_2, x, y_3, linewidth = 0.8)
        ax0.plot(x, y_0, x, y_1, x, y_2, linewidth=1.5)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.ylim((-0.25, 0.20))

        #plt.title('Velocity' + ' Magnitude')
        plt.xlabel('Time(s)', {'size': 15})
        plt.ylabel('Velocity Magnitude Error', {'size': 15})

        plt.legend(['SAE-LSTM-bunch m=7', 'SAE-LSTM-bunch m=12', 'SAE-LSTM-bunch m=35'], loc=led_location,
                   fontsize=15,ncol=n)

        plt.show()

    else:
        print(' Please check the field name.')

def rmse(ori_data, rom_data):

    rmse_value = []
    if len(ori_data) != len(rom_data):
        print('the length of these two array do not match')
    else:
        for i in range(len(rom_data)):
            value = np.sqrt(mean_squared_error(ori_data[i], rom_data[i]))
            if i == 0:
                rmse_value = value
            else:
                rmse_value = np.hstack((rmse_value,value))
        rmse_value = np.reshape(rmse_value,(-1,1))
    return rmse_value

def rmse_of_two(ori_data, rom_data):
    # , rom_data_1):
    if ori_data.ndim == rom_data.ndim:
        if rom_data.ndim == 3:
            y_u = ori_data[...,0] # u
            y_v = ori_data[...,1] # v
            y_0_u = rom_data[...,0] # u
            y_0_v = rom_data[...,1] # v
            rmse_x = rmse(y_u, y_0_u)
            rmse_y = rmse(y_v, y_0_v)
            plt.figure(1)
            plt.plot(rmse_x)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('RMSE of x axis',{'size' : 11})
            plt.figure(2)
            plt.plot(rmse_y)
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('RMSE of y axis',{'size' : 11})
            plt.show()
        elif rom_data.ndim == 2:
            rmse_value = rmse(ori_data, rom_data)
            # rmse_value_1 = rmse(ori_data, rom_data_1)
            # print(pcc.shape[0], pcc.shape[1])
            plt.figure(1)
            # x = np.linspace(0,rmse_value.shape[0], rmse_value.shape[0])
            # plt.plot(x, rmse_value, x, rmse_value_1)
            plt.plot(rmse_value)
            # plt.ylim((-0.0001, 0.02))
            plt.xlabel('Time(s)',{'size' : 11})
            plt.ylabel('RMSE',{'size' : 11})
            # plt.legend(['7', '8'], loc='lower right')   
            plt.show()
    else:
        print('the dimension of these two series are not equal. Please check them.')


# Check for Inf or -Inf and replace them with NaN
def replace_inf_with_nan(data):
    data[np.isinf(data)] = np.nan
    return data

# Interpolate to fill NaN values caused by Inf replacement
def interpolate_nan(data):
    nans, x = np.isnan(data), lambda z: z.nonzero()[0]
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data



def rmse_over_time(ori_data, rom_data_0, rom_data_1,rom_data_2,fieldName,maxValue,startNumber,loc,n):
    # Initialize lists to store Pearson correlations for each time step
    corrs_pred1 = []
    corrs_pred2 = []
    corrs_pred3 = []
    
    # Convert to numpy arrays for further processing
    corrs_pred1 = np.array(corrs_pred1)
    corrs_pred2 = np.array(corrs_pred2)
    corrs_pred3 = np.array(corrs_pred3)


    # Check for Inf or -Inf and replace them with NaN
    corrs_pred1 = replace_inf_with_nan(rom_data_0)
    corrs_pred2 = replace_inf_with_nan(rom_data_1)
    corrs_pred3 = replace_inf_with_nan(rom_data_2)

    # Interpolate to fill NaN values caused by Inf replacement
    rom_data_0 = interpolate_nan(corrs_pred1)
    rom_data_1 = interpolate_nan(corrs_pred2)
    rom_data_2 = interpolate_nan(corrs_pred3)

	
    if fieldName == 'Velocity':
        rmse_0 = rmse(ori_data, rom_data_0)
        rmse_1 = rmse(ori_data, rom_data_1)
        rmse_2 = rmse(ori_data, rom_data_2)
        #rmse_3 = rmse(ori_data, rom_data_3)
        # rmse_4 = rmse(ori_data, rom_data_4)
        # rmse_5 = rmse(ori_data, rom_data_5)
        #print(rmse_0.shape)
        # plt.figure(1)
        # x = np.linspace(5,8,600)
    
        #number=50
        rate=10
        number=14
        fig = plt.figure(figsize=(10, 7))
        fig.dpi=200
        ax = fig.add_subplot()
        #x = np.linspace(0,rmse_0.shape[0]/10,rmse_0.shape[0])
        x = np.linspace(startNumber+0, startNumber+(rmse_0.shape[0]-number) / rate, rmse_0.shape[0]-number)[1:]
        #print(x)
        ax.set_prop_cycle(color = ['#f6b93b','#BF3EFF','#104E8B','#8DEEEE'], linestyle = ['-', '-', '-', '-'])
        # x = np.linspace(0,15,2875)
        y_0 = rmse_0
        y_1 = rmse_1
        y_2 = rmse_2
        #y_3 = rmse_3

        y_0 = moving_average(rmse_0, number)[int(number/2)+1:rmse_0.shape[0]-int(number/2)]
        y_1 = moving_average(rmse_1, number)[int(number/2)+1:rmse_0.shape[0]-int(number/2)]
        y_2 = moving_average(rmse_2, number)[int(number/2)+1:rmse_0.shape[0]-int(number/2)]
        #y_2 = moving_average(rmse_2, number)
        #y_3 = moving_average(rmse_3, number)

        # y_4 = rmse_4[-1800:-1200,:]
        # y_5 = rmse_5[-1800:-1200,:]
        # plt.title('Correlation Coefficient')
        ax.plot(x, y_0, x, y_1,x, y_2,linewidth=1.5)
        # , x, rmse_4, linewidth = 0.6)
        # plt.xlim((-0.1, 200.1))# range
        #plt.ylim((-0.005, 2))
        #plt.ylim((0, 0.016))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim((0, maxValue))

        plt.xlabel('Time(s)',{'size' : 15})
        plt.ylabel('RMSE',{'size' : 15})
        # plt.xticks(np.arange(0,200.1,25))
        #plt.yticks(np.arange(0,0.016,0.002))
        plt.legend(['SAE-LSTM-bunch', 'SAE-LSTM-individual','SAE-DMD'], loc=loc,fontsize=15,ncol=n)
        plt.show()
    elif fieldName == 'U':
        point = [0,ori_data.shape[0]]
        #point = [2800,3250]
        rmse_0 = rmse(ori_data, rom_data_0)
        rmse_1 = rmse(ori_data, rom_data_1)
        rmse_2 = rmse(ori_data, rom_data_2)
        #rmse_3 = rmse(ori_data, rom_data_3)
        # rmse_4 = rmse(ori_data, rom_data_4)
        # rmse_5 = rmse(ori_data, rom_data_5)
        #print(rmse_0.shape)
        # plt.figure(1)
        # x = np.linspace(5,8,600)
        y_0=rmse_0
        y_1=rmse_1
        y_2=rmse_2
        #y_3=rmse_3
        number=14
        y_0 = moving_average(rmse_0, number)[int(number/2)+1:rmse_0.shape[0]-int(number/2)]
        y_1 = moving_average(rmse_1, number)[int(number/2)+1:rmse_0.shape[0]-int(number/2)]
        y_2 = moving_average(rmse_2, number)[int(number/2)+1:rmse_0.shape[0]-int(number/2)]
        #rate=0.1
        rate=10
        fig=plt.figure(figsize=(10, 7))
        fig.dpi=200
        ax0 = fig.add_subplot()
        #fig, ax = plt.subplots()
        #x = np.linspace(0,rmse_0.shape[0]/10,rmse_0.shape[0])
        #x = np.linspace(0,rmse_0.shape[0]/1,rmse_0.shape[0])#chzhu        
        #print(x)
        ax0.set_prop_cycle(color=['#f6b93b', '#BF3EFF', '#104E8B', '#8DEEEE'], linestyle=['-', '-', '-', '-'])
        x = np.linspace(startNumber+0, startNumber+(rmse_0.shape[0]-number) / rate, rmse_0.shape[0]-number)[1:]

        # x = np.linspace(0,15,2875)
        #y_0 = moving_average(rmse_0, number)#chzhu
        #print(y_0.shape)
        #print(y_0)
        #y_1 = moving_average(rmse_1, number)#chzhu
        #print(y_1)
        #y_2 = moving_average(rmse_2, number)#chzhu
        #print(y_2)
        #y_3 = moving_average(rmse_3, number)#chzhu
        #print(y_3)
        # y_4 = rmse_4[-1800:-1200,:]
        # y_5 = rmse_5[-1800:-1200,:]
        # plt.title('Correlation Coefficient')
        #ax.plot(x, y_0[point[0]:point[1],:], x, y_1[point[0]:point[1],:], x, y_2[point[0]:point[1],:], x, y_3[point[0]:point[1],:])
        #ax0.plot(x, y_0[point[0]:point[1]], x, y_1[point[0]:point[1]], x, y_2[point[0]:point[1]])
        ax0.plot(x, y_0, x, y_1,x, y_2,linewidth=1.5)
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # , x, rmse_4, linewidth = 0.6)
        # plt.xlim((-0.1, 200.1))# range
        #plt.ylim((-0.005, 2))
        plt.ylim((-0.0002, maxValue))

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel('Time(s)',{'size' : 15})
        plt.ylabel('RMSE',{'size' : 15})
        # plt.xticks(np.arange(0,200.1,25))
        # plt.yticks(np.arange(0,0.081,0.02))
        plt.legend(['SAE-LSTM-bunch', 'SAE-LSTM-individual','SAE-DMD'], loc=loc,fontsize=15,ncol=n)
        plt.show()



# copy original files
def copyFiles(sourceDir,targetDir):
    if sourceDir.find("exceptionfolder")>0:
        return

    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,file)
        targetFile = os.path.join(targetDir,file)

        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
                # print(targetFile+" copy succeeded")

        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)


def transform_vector(data, num, originalFolder, destinationFolder, fileName, fieldName):

    folder = os.path.exists(destinationFolder)

    if not folder: 
        print('start to create the destination folder')   
        os.makedirs(destinationFolder)       
        copyFiles(originalFolder,destinationFolder) 

    print('start to store data as a new variable')
    if len(data.shape) == 3:
        w_zero = np.zeros((data.shape[0], data.shape[1],1))
        # print(w_zero.shape)
        data=np.concatenate((data,w_zero), axis = 2)
        # print(data.shape)
    i = 0
    for i in range(num):

        f_filename = destinationFolder + fileName + str(i)+ ".vtu"
        f_file = vtktools.vtu(f_filename)
        fieldNames = f_file.GetFieldNames()
        # if fieldName in fieldNames:
        #     f_file.RemoveField(fieldName)

        if len(data[i].shape) == 1:
            f_file.AddScalarField(fieldName, data[i])
        
        elif len(data[i].shape) == 2:
            f_file.AddVectorField(fieldName, data[i])

        else:
            print('The shape of output and setted field are not matched')

        f_file.Write(f_filename)

    print('transform succeed')

# Check for Inf or -Inf and replace them with NaN
def replace_inf_with_nan(data):
    data[np.isinf(data)] = np.nan
    return data

# Interpolate to fill NaN values caused by Inf replacement
def interpolate_nan(data):
    nans, x = np.isnan(data), lambda z: z.nonzero()[0]
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data

def moving_average_2(data, window_size):
    return np.convolve(data,np.ones(window_size)/window_size,mode='valid')

       
def cc4(ori_data, rom_data_0, rom_data_1,rom_data_2,y_axis_min,fieldName,startNumber,loc,n):
    if ori_data.ndim == rom_data_0.ndim:
        if rom_data_0.ndim == 3:
            print('the dimension is equal 3.')            

        elif rom_data_0.ndim == 2:
            # Original array
            original = np.transpose(ori_data)

            num_signals = ori_data.shape[0]  # Number of snapshots
            time_series_length = ori_data.shape[0]  # Length of the time series

            # Predicted arrays (simulated predictions) - all have the same shape as the original
            predicted1 = np.transpose(rom_data_0)  
            predicted2 = np.transpose(rom_data_1)  
            predicted3 = np.transpose(rom_data_2)   

            # Initialize lists to store Pearson correlations for each time step
            corrs_pred1 = []
            corrs_pred2 = []
            corrs_pred3 = []
                                      
            # Calculate Pearson correlation for each time step (column)
            for t in range(time_series_length):
                corr1 = np.corrcoef(original[:, t], predicted1[:, t])[0, 1]
                corr2 = np.corrcoef(original[:, t], predicted2[:, t])[0, 1]
                corr3 = np.corrcoef(original[:, t], predicted3[:, t])[0, 1]
    
                corrs_pred1.append(corr1)
                corrs_pred2.append(corr2)
                corrs_pred3.append(corr3)
                    

            # Convert to numpy arrays for further processing
            corrs_pred1 = np.array(corrs_pred1)
            corrs_pred2 = np.array(corrs_pred2)
            corrs_pred3 = np.array(corrs_pred3)


            # Check for Inf or -Inf and replace them with NaN
            corrs_pred1 = replace_inf_with_nan(corrs_pred1)
            corrs_pred2 = replace_inf_with_nan(corrs_pred2)
            corrs_pred3 = replace_inf_with_nan(corrs_pred3)

            # Interpolate to fill NaN values caused by Inf replacement
            corrs_pred1 = interpolate_nan(corrs_pred1)
            corrs_pred2 = interpolate_nan(corrs_pred2)
            corrs_pred3 = interpolate_nan(corrs_pred3)
            
            fig=plt.figure(figsize=(10, 7))
            fig.dpi=200
            ax0 = fig.add_subplot()
            ax0.set_prop_cycle(color = ['#f6b93b','#BF3EFF','#104E8B','#8DEEEE'], linestyle = ['-', '-', '-', '-'])
                   
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
            plt.xlabel('Time(s)',{'size' : 15})
            plt.ylabel('Pearson Correlation Coefficient',{'size' : 15})            

             
            if fieldName == 'Velocity':
                rate=10
                windowsize=10
                x = np.linspace(startNumber+0,startNumber+ori_data.shape[0]/rate,ori_data.shape[0])
                y_0 = moving_average_2(corrs_pred1,windowsize)
                y_1 = moving_average_2(corrs_pred2,windowsize)
                y_2 = moving_average_2(corrs_pred3,windowsize)
                
                plt.ylim((y_axis_min, 1.001))
                ax0.plot(x, y_0, x, y_1,x, y_2,linewidth=1.5)
                plt.legend(['SAE-LSTM-bunch', 'SAE-LSTM-individual','SAE-DMD'], loc=loc,fontsize=15,ncol=n)
    
            
            elif fieldName == 'U':                
                rate=10
                windowsize=14

                y_0 = moving_average_2(corrs_pred1,windowsize)[int(windowsize/2)+1:ori_data.shape[0]-int(windowsize/2)]
                y_1 = moving_average_2(corrs_pred2,windowsize)[int(windowsize/2)+1:ori_data.shape[0]-int(windowsize/2)]
                y_2 = moving_average_2(corrs_pred3,windowsize)[int(windowsize/2)+1:ori_data.shape[0]-int(windowsize/2)]

                x = np.linspace(startNumber+0, startNumber+(ori_data.shape[0]-windowsize) / rate, ori_data.shape[0]-windowsize)[7:]
                #x = np.linspace(startNumber+0,startNumber+ori_data.shape[0]/rate,ori_data.shape[0])
                
                plt.ylim((y_axis_min, 1.006))
                ax0.plot(x, y_0, x, y_1,x, y_2,linewidth=1.5)                
                plt.xlabel('Time(s)',{'size' : 15})
                plt.legend(['SAE-LSTM-bunch', 'SAE-LSTM-individual','SAE-DMD'], loc=loc,fontsize=15,ncol=n)
            plt.show()
    else:
        print('the dimension of these two series are not equal. Please check them.')



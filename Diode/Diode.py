import os
import time

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import numpy as np
from math import log
from sklearn.metrics import mean_squared_error


from pyrcn.echo_state_network import ESNRegressor

#Save dataset in dataArray

startt = time.time()
inter = 4 #interpolation amount
length=101
    
dataArray = []
files= 0
for filename in os.listdir("Train"):
        fname = "Train/" + filename
        with open(fname) as f:
            files+=1
            for line in f: # read rest of lines
                    temp = [float(x) for x in line.split()]
                    app =[temp[0],temp[1],0.06,temp[2]]
                    dataArray.append(app)
        
lengths = 0
if(lengths == 1):
    for filename in os.listdir("L_08"):
             fname = "L_08/" + filename
             with open(fname) as f:
                 files+=1
                 for line in f: # read rest of lines
                         temp = [float(x) for x in line.split()]
                         app =[temp[0],temp[1],0.08,temp[2]]
                         dataArray.append(app)

dataArray = np.array(dataArray)
trainarrayx = dataArray[:,:3]
trainarrayy =  dataArray[:,3:]

i=0
for x in trainarrayy:
      trainarrayy[i] = log(x)
      i+=1
    

    
newtrainy = []
for i in range(0,files):
    interp_func2 = interp1d(trainarrayx[i*(length+1):(i+1)*length,0] , trainarrayy[i*(length+1):(i+1)*length,0],bounds_error=False,  fill_value="extrapolate" )
    newarr = interp_func2(np.arange(0, 0.600001,(1/inter)*(trainarrayx[1,0] - trainarrayx[0,0])))   
    newtrainy.extend(newarr.copy())
        
newtrainy = np.array(newtrainy)

newtrainx = []
for i in range(0,files):
        interp_func2 = interp1d(trainarrayx[i*(length+1):(i+1)*length,0] , trainarrayx[i*(length+1):(i+1)*length,1],bounds_error=False,  fill_value="extrapolate" )
        newarr2 = interp_func2(np.arange(0, 0.600001,(1/inter)*(trainarrayx[1,0] - trainarrayx[0,0])))   
        temptrainx= np.vstack((np.arange(0, 0.600001,(1/inter)*(trainarrayx[1,0] - trainarrayx[0,0])), newarr2)).T
        newtrainx.extend(temptrainx.copy())
    
newtrainx = np.array(newtrainx)

dataArray2 = []
files= 0
for filename in os.listdir("Test"):
        fname = "Test/" + filename
        with open(fname) as f:
            files += 1
            for line in f: # read rest of lines
                    temp = [float(x) for x in line.split()]
                    app =[temp[0],temp[1],0.06,temp[2]]
                    dataArray2.append(app)
    
dataArray2 = np.array(dataArray2)
testarrayx = dataArray2[:,:3]
testarrayy =  dataArray2[:,3:]


i=0
for x in testarrayy:
      testarrayy[i] = log(x)
      i+=1
 


newtesty = []
for i in range(0,files):
        interp_func1 = interp1d(testarrayx[i*(length+1):(i+1)*length,0] , testarrayy[i*(length+1):(i+1)*length,0],bounds_error=False,  fill_value="extrapolate" )
        newarr = interp_func1(np.arange(0, 0.600001,(1/inter)*(testarrayx[1,0] - testarrayx[0,0])))   

        newtesty.extend(newarr.copy())


newtestx = []
for i in range(0,files):
        interp_func2 = interp1d(testarrayx[i*(length+1):(i+1)*length,0] , testarrayx[i*(length+1):(i+1)*length,1],bounds_error=False,  fill_value="extrapolate" )
        newarr2 = interp_func2(np.arange(0, 0.600001,(1/inter)*(testarrayx[1,0] - testarrayx[0,0])))   
        temptrainx= np.vstack((np.arange(0, 0.600001,(1/inter)*(testarrayx[1,0] - testarrayx[0,0])), newarr2)).T
        newtestx.extend(temptrainx.copy())
  
newtestx = np.array(newtestx)

start = time.time()
reg = ESNRegressor(spectral_radius = 0.99,sparsity = 0.3, hidden_layer_size = 3000)
end = time.time()

reg.fit(X=newtrainx[:], y=newtrainy)
        
y_pred = reg.predict(newtestx[:])  # output is the prediction for each input example

        
plt.figure(0)

plt.plot(newtestx[:,0],y_pred[:], color = 'blue' ) 
plt.plot(newtestx[:,0],newtesty[:], color='red')
        

    
        
mse = mean_squared_error(newtesty[:],y_pred[:])
endtt=time.time()
print("str(k) + "" MSE for dataset "": " + str(mse) + "Execution time:" + str(endtt - startt) + "seconds")
    ############## 

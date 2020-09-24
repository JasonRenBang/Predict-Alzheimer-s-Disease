# -*- coding: utf-8 -*-
"""
@author: Jason Chen
"""

import numpy as np
import pandas as pd

#read files
class ReadFilesName():
    def __init__(self, path):
        self.path = path
        
    #read files
    def ReadFileName(self):
        dataset = pd.read_csv(self.path,index_col = 0)        
        return dataset
    
    #print dataset
    def printData(self,dataset):
        print("print dataset:")
        print(dataset.head)

#preprocess       
class Preprocess():
    def __init__(self, dataset):
       
        self.dataset = pd.DataFrame(dataset)
        
    #delete column
    def deleteColumn(self,columnName):
        
        self.dataset = self.dataset.drop(columnName, axis=1, inplace = False)
        return self.dataset
    
    #delete nan values
    def deleteNaN(self):
        self.dataset = self.fillna(0, inplace = False)
        
        return self.dataset
    
    #get data
    def getdata(self):
        
        return self.dataset
    
    #print data
    def printData(self):
        self.dataset.set_option('display.max_rows', None)
        print (self.dataset)
        
# classify the data
def stumpClassify(data_matrix, dimen, thresh_val, thresh_ineq):  
    
    # compare threshold, change to -1 and 1
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


# a werk learner
def buildStump(data_arr, class_labels, d):
    
    data_matrix = np.mat(data_arr)
    label_mat = np.mat(class_labels).T  
    m, n = np.shape(data_matrix)
    num_steps = 10.0 
    best_stump = {}  
    best_clas_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf 
    
    for i in range(n): 
        range_min = data_matrix[:, i].min()  
        range_max = data_matrix[:, i].max()
        
        # set step
        step_size = (range_max - range_min) / num_steps  
        
        #access all features
        # loop over all range in current dimension
        
        for j in range(-1, int(num_steps) + 1):   
            
            # go over less than and greater than
            for inequal in ['lt', 'gt']:  
                thresh_val = (range_min + float(j) * step_size)  
                
                # call stump classify with i, j, lessThan
                predicted_vals = stumpClassify(data_matrix, i, thresh_val, inequal)  
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0  
                
                # calc total error multiplied by D
                weighted_error = d.T * err_arr 

                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                #     i, thresh_val, inequal, weighted_error))
                if weighted_error < min_error:
                    
                    min_error = weighted_error
                    best_clas_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_clas_est

#Adaboost model
def Adaboost(data_arr, class_labels, num_iter):
   
    weak_class_arr = []  
    m = np.shape(data_arr)[0]
    
    # init D to all equal  
    d = np.mat(np.ones((m, 1)) / m)  
    agg_class_est = np.mat(np.zeros((m, 1)))  
    for i in range(num_iter):
        
        # build Stump
        best_stump, error, class_est = buildStump(data_arr, class_labels, d) 
        
        # calc alpha, throw in max(error,eps) to account for error=0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16))) 
        
        best_stump['alpha'] = alpha
        
        # store Stump Params in Array
        weak_class_arr.append(best_stump)  
        
        # exponent for D calc, getting messy
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)  
        
        # Calc New D for next iteration
        d = np.multiply(d, np.exp(expon))         
        d = d / d.sum()
        
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        agg_class_est += alpha * class_est
        
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        
        if error_rate == 0.0: break  
    return weak_class_arr, agg_class_est



def adaClassify(dat2class, classifier_arr):
    
    # do stuff similar to last aggClassEst in adaBoostTrainDS
    data_matrix = np.mat(dat2class)  
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))  
    for i in range(len(classifier_arr)): 
        
        # call stump classify
        class_est = stumpClassify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'], classifier_arr[i]['ineq'])  
        
        agg_class_est += classifier_arr[i]['alpha'] * class_est
    return np.sign(agg_class_est)



if __name__ == '__main__':
    
    #read files
    path = 'C:\\Users\\Jason Chen\\Desktop\\BLSTMtest\\PET_AV45_NoNANWithReduceLabel.csv'
    readfile = ReadFilesName(path)
    dataset = readfile.ReadFileName()
    readfile.printData(dataset)
    
    #preprocess dataset
    pre = Preprocess(dataset) 
    pre = pre.getdata()
    print(pre.head)
    pre = pre.drop('VISCODE', axis = 1, inplace =False)
    pre = pre.drop('DX_bl', axis = 1, inplace =False)
    pre = pre.drop('DXCHANGE', axis = 1, inplace =False)
    pre = pre.drop('AGE', axis = 1, inplace =False)
    pre = pre.drop('DX', axis = 1, inplace =False)   
    pre = pre.fillna(0, inplace = False)
    dataset= pre
    
    values = dataset.values
    value = values.astype('float32')
    values = values.astype('int64')
    
    #create features and label
    data = []
    label = []
    
    dataa = value[:,:]
    data = values[:,:]
    
    
    
    for i in range(len(data)):
        if (data[i,0]>=27):
            label.append(1.0)
        else:
            label.append(-1.0)
        i=i+1
    
    label=np.array(label)
    
    size2= int(len(dataa)*0.8)
    print(dataa)
        
    train_x = dataa[0:size2,:]
    train_y =label[0:size2]
    print("train_x",np.shape(train_x))
    print("train_y",np.shape(train_y))
    test_x = dataa[size2:len(dataa),:]
    print("test_x",np.shape(test_x))
    test_y = label[size2:len(dataa)]
    print("test_y",np.shape(test_y))
    
    print("")
    print("train_x")
    print(train_x)
    print("")
    print("train_y")
    print(train_y)
    print("")
    print("test_x")
    print(test_x)
    print("")
    print("test_y")
    print(test_y)
    print("")
    
    classifier_array, agg_class_est = Adaboost(train_x, train_y,40)

    # get prediction
  
    prediction = adaClassify(test_x, classifier_array)

    err_arr = np.mat(np.ones([424, 1]))
    err_num = err_arr[prediction != np.mat(test_y).T].sum()
    print("accuracy:%.3f" % (1 - err_num / float(424)))

    



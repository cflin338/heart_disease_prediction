# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:16:59 2020

@author: clin4
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def unique_listings(x, y, scale_factor):
    #designed for points where data is discrete 
    #calc weights for each unique x value, will be used to scale unique point counts
    unique_x = np.sort(np.unique(x))
    x_weights = {}
    for i in unique_x:
        x_weights[i] = list(x).count(i)
    
    #calc weights for each unique point
    weights = []
    unique_list = []
    
    for (i,j) in zip(x,y):
        if [i,j] in unique_list:
            weights[unique_list.index([i,j])] = weights[unique_list.index([i,j])] + 1
        else:
            unique_list.append([i,j])
            weights.append(1)
    
    for i in range(len(unique_list)):
        weights[i] = weights[i] / x_weights[unique_list[i][0]] * scale_factor
    
    return np.array(unique_list), np.array(weights)

variables = ['age','sex','cp','restbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
#read csv and placing data into matrix/ arrays data_input, data_output

file = 'C:\\Users\\clin4\\Documents\\py\\heart_disease\\archive\\heart.csv'

file = 'C:\\Users\\clin4\\Downloads\\processed.cleveland.data'

data_input = []
bad_input = []
data_output = []
#output_vals = [list(i).index(1) for i in data_output]
count = 0
distinctions = {0: [], 1: [], 2: [], 3: [], 4: []}
with open(file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        try:
            data_input.append([float(i) for i in row[0:-1]])
            cat = int(row[-1])
            v = [0]*5
            v[cat] = 1
            distinctions[cat].append(data_input[-1])
            data_output.append(v)
            
        except ValueError:
            bad_input.append(row)
        count = count + 1
output_vals = np.array([list(i).index(1) for i in data_output])

data_input = np.array(data_input)
data_output = np.array(data_output)
point_count = len(data_output)

zero_points = np.array([data_input[i] for i in range(point_count) if data_output[i][0]==1])
#160
some_points = np.array([data_input[i] for i in range(point_count) if data_output[i][0]!=1])
#137
one_points = np.array([data_input[i] for i in range(point_count) if data_output[i][1]==1])
#54
two_points = np.array([data_input[i] for i in range(point_count) if data_output[i][2]==1])
#35
three_points = np.array([data_input[i] for i in range(point_count) if data_output[i][3]==1])
#35
four_points = np.array([data_input[i] for i in range(point_count) if data_output[i][4]==1])
#13



print('{} data point, {} data points with a ? value'.format(len(data_input), len(bad_input)))

#### --------------------------------------------basic visualization----------------------------------------------------------------#######
plt.close('all')
#distribution of all cardiovascular levels for 'population' baseline
plt.figure()
plt.title('overall population distribution')
plt.hist(output_vals, bins = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], )
plt.xlabel('Heart Disease Presence')

plt.figure()
#generic visualization
for i in range(13):
    plt.subplot(5,3,i+1)
    plt.title('{} v Heart Disease Presence, Generic Visualization'.format(variables[i]))
    plt.scatter(data_input[:,i], output_vals)
    plt.ylabel('Heart Disease Presence')
    plt.xlabel(variables[i])

#visualization for plots with few unique xvals: 1, 2, 5, 6, 8, 10, 11, 12
#       sex, cp, fbs, 
discrete_categories = [1, 2, 5, 6, 8, 10, 11, 12]
plt.figure()
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.title('{} vs Heart disease presence, weighted points'.format(variables[discrete_categories[i]]))
    u,w = unique_listings(data_input[:,discrete_categories[i]], output_vals, 1000)
    plt.scatter(u[:,0], u[:,1], s = w)
    plt.ylabel('Heart Disease Presence')
    plt.xlabel(variables[discrete_categories[i]])

#visualization for plots with continuous data points: 0, 3, 4, 7, 9
plt.figure()
continuous_categories = [0, 3, 4, 7, 9]
for i in range(len(continuous_categories)):
    plt.subplot(2,3,i+1)
    plt.title('{} vs Heart disease presence, boxplot'.format(variables[continuous_categories[i]]))
    tempX = data_input[:,continuous_categories[i]]
    zeros = [tempX[i] for i in range(len(output_vals)) if output_vals[i]==0]
    ones = [tempX[i] for i in range(len(output_vals)) if output_vals[i]==1]
    twos = [tempX[i] for i in range(len(output_vals)) if output_vals[i]==2]
    threes = [tempX[i] for i in range(len(output_vals)) if output_vals[i]==3]
    fours = [tempX[i] for i in range(len(output_vals)) if output_vals[i]==4]
    plt.boxplot(np.array([zeros, ones, twos, threes, fours],dtype = object), positions = [0, 1, 2, 3, 4])
    plt.xlabel('Heart Disease Presence')
    plt.ylabel(variables[continuous_categories[i]])

    
###### --------------------------------------------2D Visualization----------------------------------------------------------------#######
#test
plt.figure();
for i in range(4):
    for j in range(i+1,5):
        plt.subplot(4,4, (i)*4+(j))
        p1 = continuous_categories[i]
        p2 = continuous_categories[j]
        plt.scatter(zero_points[:,p1], zero_points[:,p2], s = 25, marker = 'o', c = 'r', label = 'No Heart Disease Present')
        plt.scatter(some_points[:,p1], some_points[:,p2], s = 15, marker = 'o', c = 'k', label = 'Some Heart Disease Present') 
        plt.xlabel(variables[p1])
        plt.ylabel(variables[p2])
        plt.legend()

plt.figure();
for i in range(4):
    for j in range(i+1,5):
        plt.subplot(4,4, (i)*4+(j))
        p1 = continuous_categories[i]
        p2 = continuous_categories[j]
        plt.scatter(zero_points[:,p1], zero_points[:,p2], s = 40, marker = 'o', c = 'r')
        plt.scatter(one_points[:,p1], one_points[:,p2], s = 30, marker = 'o', c = 'k')
#        plt.scatter(two_points[:,p1], two_points[:,p2], s = 75, marker = 'o', c = 'b')
        plt.scatter(three_points[:,p1], three_points[:,p2], s = 20, marker = 'o', c = 'b')
        plt.scatter(four_points[:,p1], four_points[:,p2], s = 10, marker = 'o', c = 'g')
        plt.xlabel(variables[p1])
        plt.ylabel(variables[p2])
        #plt.legend()

#-------------------------Binary Categorization----------------------------------------
#visualization for plots with continuous data points: 0, 3, 4, 7, 9
plt.figure()
continuous_categories = [0, 3, 4, 7, 9]
for i in range(len(continuous_categories)):
    plt.subplot(5,2,2*i+1)
    plt.title('{} vs Heart disease presence, boxplot'.format(variables[continuous_categories[i]]))
    tempX = data_input[:,continuous_categories[i]]
    z = zero_points[:,continuous_categories[i]]#[tempX[i] for i in range(len(output_vals)) if output_vals[i]==0]
    s = some_points[:, continuous_categories[i]]#[tempX[i] for i in range(len(output_vals)) if output_vals[i]>0]
    
    plt.boxplot(np.array([z, s],dtype = object), positions = [0, 1])
    plt.xlabel('Heart Disease Presence')
    plt.ylabel(variables[continuous_categories[i]])
    
    plt.subplot(5,2,2*i+2)
    sns.kdeplot(z, color = 'r', shade = True, label = 'No Disease')
    sns.kdeplot(s, color = 'b', shade = True, label = 'Some Disease')
    plt.legend()
    
 #visualization of correlation
plt.figure()
#plt.subplot(1,2,1)
corr_matrix = np.zeros((13,13))
for i in range(len(variables)-1):
    for j in range(i, len(variables)-1):
        c = np.corrcoef(zero_points[:,i], zero_points[:,j])
        corr_matrix[i][j] = c[0][1]
sns.heatmap(corr_matrix, annot = True, cmap='RdYlGn',linewidths=0.2,annot_kws={'size':12})
plt.title('correlation heatmap with no disease present')
#plt.subplot(1,2,2)
plt.figure()
plt.title('correlation heatmap with some disease present')
corr_matrix = np.zeros((13,13))
for i in range(len(variables)-1):
    for j in range(i, len(variables)-1):
        c = np.corrcoef(some_points[:,i], some_points[:,j])
        corr_matrix[i][j] = c[0][1]
sns.heatmap(corr_matrix, annot = True, cmap='RdYlGn',linewidths=0.2,annot_kws={'size':12})



# K-NN evaluation
verbose = False

import tensorflow as tf
selected_variables = [0,1,6,7,8,9,10,11,12]
selected_data_input = data_input[:,selected_variables]

knn_range = 30
range_min = 1
error_list = np.zeros((knn_range,2))
print('kNN model test for predicting range 0-4')
train_indices = np.random.choice(len(selected_data_input), round(len(selected_data_input) * 0.8), replace=False)
test_indices =np.array(list(set(range(len(selected_data_input))) - set(train_indices)))
data_input_train = selected_data_input[train_indices]
data_output_train = data_output[train_indices]
data_input_test = selected_data_input[test_indices]
data_output_test = data_output[test_indices]
for k in range(range_min, range_min+knn_range):
    
    feature_number = len(selected_variables)
    categorization_number = 5
    #k = 10
    
    # model setup
    tf.compat.v1.disable_eager_execution()
    
    x_data_train = tf.compat.v1.placeholder(shape=[None, feature_number], dtype=tf.float32)
    y_data_train = tf.compat.v1.placeholder(shape=[None, categorization_number], dtype=tf.float32)
    x_data_test = tf.compat.v1.placeholder(shape=[None, feature_number], dtype=tf.float32)
    
    # manhattan distance
    #distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)
    distance = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(x_data_train, tf.expand_dims(x_data_test, 1)),axis = 2))
    # nearest k points
    _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    top_k_label = tf.gather(y_data_train, top_k_indices)
    
    sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
    prediction = tf.argmax(sum_up_predictions, axis=1)
    
    # model training
    
    sess = tf.compat.v1.Session()
    prediction_outcome = sess.run(prediction, feed_dict={x_data_train: data_input_train,
                                   x_data_test: data_input_test,
                                   y_data_train: data_output_train})
    
    #test model
    accuracy = 0
    for pred, actual in zip(prediction_outcome, data_output_test):
        if pred == np.argmax(actual):
            accuracy += 1
    if verbose:
        print('{}% accuracy with {} neighbors'.format(round(100*accuracy / len(prediction_outcome),2), k))
    error_list[k-range_min,:] = np.array([k, accuracy/len(prediction_outcome)])
    
print('Max accuracy for {} neighbors with {}% accuracy'.format(np.argmax(error_list[:,1])+5, np.round(100*np.max(error_list[:,1]),2)))

#------------KNN evaluation, binary output----------------
print('kNN model test for predicting binary values')
binary_outputs = np.zeros((len(output_vals),2))
for i in range(len(output_vals)):
    if output_vals[i] >= 1:
        binary_outputs[i,1] = 1
    else:
        binary_outputs[i,0] = 1
        
data_output_train = binary_outputs[train_indices]
data_output_test = binary_outputs[test_indices]
pcheck_list = []
for k in range(range_min, range_min+knn_range):
    
    feature_number = len(selected_variables)
    categorization_number = 2
    #k = 10
    
    # model setup
    tf.compat.v1.disable_eager_execution()
    
    x_data_train = tf.compat.v1.placeholder(shape=[None, feature_number], dtype=tf.float32)
    y_data_train = tf.compat.v1.placeholder(shape=[None, categorization_number], dtype=tf.float32)
    x_data_test = tf.compat.v1.placeholder(shape=[None, feature_number], dtype=tf.float32)
    
    # manhattan distance
#    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)
    distance = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(x_data_train, tf.expand_dims(x_data_test, 1)),axis = 2))
    # nearest k points
    _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    top_k_label = tf.gather(y_data_train, top_k_indices)
    
    sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
    prediction = tf.argmax(sum_up_predictions, axis=1)
    
    # model training
    
    sess = tf.compat.v1.Session()
    prediction_outcome = sess.run(prediction, feed_dict={x_data_train: data_input_train,
                                   x_data_test: data_input_test,
                                   y_data_train: data_output_train})
    
    #test model
    accuracy = 0
    positive_check = np.zeros((2,2))
    for pred, actual in zip(prediction_outcome, data_output_test):
        if pred == np.argmax(actual):
            accuracy += 1
        positive_check[pred, np.argmax(actual)] += 1
    if verbose:
        print('{}% accuracy with {} neighbors'.format(round(100*accuracy / len(prediction_outcome),2), k))
    pcheck_list.append(positive_check)
    error_list[k-range_min,:] = np.array([k, accuracy/len(prediction_outcome)])
    
partial_set = []    

print('Max accuracy for {} neighbors with {}% accuracy'.format(np.argmax(error_list[:,1])+5, np.round(100*np.max(error_list[:,1]),2)))
print('{} false positives and {} false negatives out of {} test points'.format(pcheck_list[np.argmax(error_list[:,1])][1,0], pcheck_list[np.argmax(error_list[:,1])][0,1], len(data_output_test)))

import pandas as pd #Import the pandas library
    
data = pd.read_csv("/home/soft25/Downloads/Churn_Modelling.csv") #Read the dataset
    
data.columns #Findout the dataset columns
    
x = data.iloc[:,3:13].values #Split the x value from the dataset
y = data.iloc[:,13].values #Split the y value from the dataset
    
data['Geography'].value_counts() #find out the count value of the Geography catageries
data['Gender'].value_counts() #find out the count value od the gender catageries
    
from sklearn.preprocessing import LabelEncoder #Import the LAbelencoder library
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1]) #Convert the categerical value into numerical
x[:,2] = le.fit_transform(x[:,2]) #Convert the categerical; value into numearical
    
from sklearn.model_selection import train_test_split #import thr train test module
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0) #Split the data into train and test
    
from sklearn.preprocessing import StandardScaler #Import the satnderdscaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #Normalizing the x_train data
x_test = sc.fit_transform(x_test) #Normalizing

#import numpy as np
#x_train,x_test = np.array(x_train,dtype=np.float32),np.array(x_test,dtype=np.float32)
#x_train.shape

import tensorflow as tf #Import the tensorflow library

training = tf.data.Dataset.from_tensor_slices((x_train,y_train)) #Convert the data numpy into tensor slices 
training = training.repeat().batch(32).prefetch(1) #Create a batch size for our data

n_dim = 10 #declare some parameters for our model
n_hidden1 = 60
n_hidden2 = 60
out = 1
learning_rate = 0.0001

initilaizer = tf.initializers.glorot_uniform() #Create a initializer for initialize weights and biases
weights = [ #Initialize the weights
    tf.Variable(initilaizer(shape=[n_dim,n_hidden1])), #Initialize weight the first layer of our model
    tf.Variable(initilaizer(shape=[n_hidden1,n_hidden2])), ##Initialize weight the hidden layer of our model
    tf.Variable(initilaizer(shape=[n_hidden2,out])), #Initialize weight the output layer of the model
                ]
biases = [
    tf.Variable(initilaizer(shape=[n_hidden1])), #Initialize biases the first layer of our model
    tf.Variable(initilaizer(shape=[n_hidden2])),#Initialize biases the hidden layer of our model
    tf.Variable(initilaizer(shape=[out])) #Initialize weight the output layer of our model
    ]

def model(x): #Define ou model
    layer_1 = tf.add(tf.matmul(x,weights[0]),biases[0]) #Create a input layer
    layer_1 = tf.nn.relu(layer_1) #Activation function for the input layer
    
    layer_2 = tf.add(tf.matmul(layer_1,weights[1]),biases[1]) #Create a hidden layer
    layer_2 = tf.nn.relu(layer_2) #Activation function for the hidden layer
    
    layer_3 = tf.add(tf.matmul(layer_2,weights[2]),biases[2]) #Create a output layer
    input_tensor = tf.cast(layer_3, dtype=tf.float32)
    return tf.nn.sigmoid(input_tensor) #Activation function for the output layer
print(model)

optimizer = tf.optimizers.Adam(learning_rate) #Initialize the oiptimizer for our model

def calculate_loss(y_pred,y_true): #Create a loss function to calculate the loss of our model
    y_pred = tf.clip_by_value(y_pred,1e-9,1.)
    return tf.reduce_mean(tf.losses.binary_crossentropy(y_true,y_pred))

def acc(y_pred,y_true): #Create a accuracy function for our model.
    correct = tf.equal(tf.cast(y_pred>0.5,tf.int64),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correct,tf.float64))

def train_step(x,y): #Create a train_sted of our model
    with tf.GradientTape() as tape:
        pred = model(x) #Call our model
        loss = calculate_loss(pred,y) #Calculate the loss of predict and true value
        
        variables = weights + biases #Adding the weights and biases
        grad = tape.gradient(loss,variables)

    optimizer.apply_gradients(zip(grad,variables))
    return pred,loss

for epoch in range(100):   #Make a iteration of our model
    accuracy_history = []
    loss_history = []
    for step,(batch_x,batch_y) in enumerate(training.take(x_train.shape[0]//32),1): #Split the data into dependent and independent variable
      x_batch = tf.cast(batch_x, dtype=tf.float32) #Convert the data type of our x 
      y_batch = tf.cast(batch_y, dtype=tf.float32) #Convert the data type of our y
      train_step(x_batch,y_batch) #train our model with x and y 
      pred = model(x_batch) #Predict the value with the help of our model
      loss = calculate_loss(pred, batch_y)
      accuracy = acc(pred,batch_y)
      print(float(accuracy),float(loss))
      accuracy_history.append(accuracy)
      loss_history.append(loss)
      
import numpy as np
acc = np.mean(accuracy_history) #Mean of our accuracy
loss1 = np.mean(loss_history) #Loss value of our model
      
print('Accuracy of the model : ',acc*100) #Print the accuracy value
print('Loss of the model :',loss1) #Print the loss value


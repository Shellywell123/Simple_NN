import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import numpy as np
import cv2 as cv
import scipy
import h5py
import os

###########################################
# Start of class
###########################################

class Simple_NN():
    """
    first go at making a class for building Nueral Networks's from scratch
    """

    def __init__(self):
        """
        class initialiser
        """
        pass

    
    def sigmoid(self,z):
        """
        activation function for binary outout between 0 and 1
        - use tanh for output between -1 and 1
        """
        s = 1/(1+np.exp(-z))
        return s

    def relu(self,z):
        """
        linear activator
        """
        r=max(0,x)
        return r

    def unpack_image_data(self,image):
        """
        INCOMPLETE
        """
        archive = h5py.File('test.h5', 'w')
        archive.create_dataset('/array', data=array)
        archive.close()
                
        return 0

    def download_training_data(self,num_of_images_per_category,list_of_categories):
        """
        uses bing_image_downloader==1-0-2 to download images 
        """
        from bing_image_downloader import downloader
        
        # put list of objects here

        lim        = str(num_of_images_per_category)
        categories = list_of_categories

        for category in categories:
            downloader.download(category, limit=lim, adult_filter_off=True, force_replace=False)

        #######################################################

        #summary of downloads
        for category in categories:
            print ('{} Images of {}'.format(lim,pet))


    def resize_image(self,image,size):
        """
        resize an image
        """
        img = Image.open(image)
        img = img.resize((size,size))
        arr = np.array(img)
        
        #flatten 
        arr = arr.flatten()
        arr=arr.reshape((size*size*3),-1)
        return arr

    def show_image(self,image):
        """
        simple function to display an image
        """
        im = Image.open(image)  
        im.show()


    def format_training_data_as_h5(self,training_data_dir):
        """
        converts images to a vector using hpy5
        """
        pixel_dim=64
        X_training_data = np.empty((pixel_dim*pixel_dim*3,1))
        Y_training_data = np.empty((1,1))

        #print(X_training_data,Y_training_data)

        #key = {}

        for subdir in os.listdir(training_data_dir):
            # print name and size of each dataset
            print (subdir,len(os.listdir(training_data_dir+'/'+subdir)))

            index=os.listdir(training_data_dir).index(subdir)
            # key[index] = subdir
            
            for image in os.listdir(training_data_dir+'/'+subdir):    

                #if statment is only here as I didnt want to train 10k images so i limited it per category
                if os.listdir(training_data_dir+'/'+subdir).index(image) >= 1000:
                    break
                else:            
                    x = self.resize_image(training_data_dir+'/'+subdir+'/'+image,pixel_dim)
                    X_training_data = np.column_stack((X_training_data, x))
                    y = np.array([index]).T
                    Y_training_data = np.column_stack((Y_training_data, y))
                    
                
        #deletes first empty data set that was created when the array was initalised
        X_training_data = np.delete(X_training_data,0,1)
        Y_training_data = np.delete(Y_training_data,0,1)

        #make image data greyscale
        X_training_data = X_training_data/255.
        Y_training_data = Y_training_data/255.

        #print(X_training_data,X_training_data.shape)
        
        hf = h5py.File('training_data.h5', 'w')
        hf.create_dataset('X_train',data=X_training_data)
        hf.create_dataset('Y_train',data=Y_training_data)
        hf.close()

    def get_key(self,training_data_dir):
        """
        gains dictionary to put catergories to indexes
        -currently gathered from dir structure of your traingin data,
        -want to store in h5 file eventually 
        """
        key = {}
        for subdir in os.listdir(training_data_dir):
            # print name and size of each dataset
            index=os.listdir(training_data_dir).index(subdir)
            key[index] = subdir
        return key

    def load_h5_training_data(self,filename):
        """
        unpacks h5 data files
        """
        hf = h5py.File(filename, 'r')
        X_training_data = np.array(hf.get('X_train'))
        Y_training_data = np.array(hf.get('Y_train'))
        key_str         = str(hf.get('Key'))
        key             = eval(key_str)

        return X_training_data,Y_training_data

    def initalize_params_0(self,w_dim):
        """
        initialises w,b to be zero
        """
        w = np.zeros((w_dim,1))
        b = 0
        return w,b

    def cost_function(self,X,Y,A):
        """
        computes the cost
        """
        m = X.shape[1]
        cost = (-1/m)*(np.sum((Y*np.log(A))+(1-Y)*np.log(1-A))) 
        return cost

    def forward_backward_prop(self,X,Y,w,b):
        """
        carries out forward and backwards propagation 
        """
        m = X.shape[1]

        #forward prop
        z    = np.dot(w.T,X)+b
        A    = self.sigmoid(z)
        cost = self.cost_function(X,Y,A)

        #back prop
        dw    = (1/m)*np.dot(X,((A-Y).T))
        db    = (1/m)*np.sum((A-Y))

        #save gradients for future use
        gradients = {"dw": dw,"db": db}

        return cost,gradients

    def gradient_descent(self,X,Y,w,b,num_iterations,learning_rate):
        """
        uses propagation in loop to optimize paramaters
        """

        #keep cost values to plot 
        costs = []
    
        for i in range(num_iterations):
        
         
            cost, gradients = self.forward_backward_prop(X,Y,w,b)
            
            dw = gradients["dw"]
            db = gradients["db"]
        
            # update params using the learning rate
            w = w - learning_rate*dw
            b = b - learning_rate*db
       
        
            # save costs values every 100 vals
            if i % 100 == 0:
                costs.append([i,cost])
        
    
            #save prarams and gradients for future use
            params = {"w": w,"b": b}
            gradients = {"dw": dw,"db": db}

        plt.plot(cost[0],costs[1])
        plt.xlabel('num or num_iterations')
        plt.ylabel('cost')
        plt.savefig('cost.png')
        
        return params, gradients, costs


    def preidict_output(self,X,w,b):
        """
        predicts binary label
        """
        m = X.shape[1]

        #initalise output as 0
        Y_prediction = np.zeros((1,m))

        z = np.dot(w.T,X)+b
        A = self.sigmoid(z)

        #round values to be binary 
        for i in range(A.shape[1]):

            if A[0][i] <= 0.5:
                Y_prediction[0][i] = 0

            if A[0][i] > 0.5:
                Y_prediction[0][i] = 1

        return Y_prediction


    def model(self,X_train,Y_train,num_iterations=2000,learning_rate=0.5):
        """
        will train model
        """
        w_dim = X_train.shape[0]
        w,b = self.initalize_params_0(w_dim)

        parameters,gradients,costs = self.gradient_descent(X_train,Y_train,w,b,num_iterations,learning_rate)
        
        w = parameters["w"]
        b = parameters["b"]

        Y_prediction_train = self.preidict_output(X_train,w,b)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

        return w,b

    def test_image(self,image,w,b,image_category):
        """
        test a model with an image

        """
        pixel_dim=64
        X_training_data = np.empty((pixel_dim*pixel_dim*3,1))
        Y_training_data = np.empty((1,1))

        x_test = self.resize_image(image,pixel_dim)/255
        key = self.get_key('dataset/bing')

        y_val  = list(key.keys())[list(key.values()).index(image_category)]
        y_test = np.array([y_val]).T

        print('#'*15)
        Y_prediction_test = self.preidict_output(x_test,w,b)
        acc = 100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100
        print("test image predicted as {} with {} % accuracy".format(key[int(np.squeeze(Y_prediction_test))],acc))
        self.show_image(image)

###########################################
# End of Class
###########################################
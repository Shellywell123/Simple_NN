import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import numpy as np
import cv2 as cv
import scipy
import h5py
import os

bing_fork_location = '/mnt/c/Users/benja/Documents/Programming/Python Projects/bing_image_downloader'

###########################################
# Start of class
###########################################

class Simple_NN():
    """
    first go at making a class for building Nueral Networks's from scratch
    """

    def __init__(self):
        """
        class initialiser yet to do anything
        """
        pass

    def download_training_data(self,num_of_images_per_category,list_of_categories):
        """
        uses bing_image_downloader==1-0-2 to download images 
        """
        
        sys.path.append(bing_fork_location)
        from bing_image_downloader import downloader
        
        lim        = str(num_of_images_per_category)
        categories = list_of_categories

        for category in categories:
            downloader.download(category, limit=lim, adult_filter_off=True, force_replace=False)
            #summary of downloads
            print ('{} Images of {}'.format(lim,pet))

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
    
    def format_training_data_as_h5(self,training_data_dir,filename):
        """
        converts images to a vector using hpy5
        """
        print('loading training data ...')
        pixel_dim=64
        X_training_data = np.empty((pixel_dim*pixel_dim*3,1))
        Y_training_data = np.empty((1,1))

        #print(X_training_data,Y_training_data)
        #key = {}

        if not os.path.exists('h5_files'):
            os.makedirs('h5_files')

        for subdir in os.listdir(training_data_dir):
            # print name and size of each dataset
            print (subdir,len(os.listdir(training_data_dir+'/'+subdir)))

            index = os.listdir(training_data_dir).index(subdir)
            # key[index] = subdir
            
            for image in os.listdir(training_data_dir+'/'+subdir):    
                print(os.listdir(training_data_dir+'/'+subdir).index(image))

                #if statment is only here as I didnt want to train 10k images so i limited it per category
                if os.listdir(training_data_dir+'/'+subdir).index(image) >= 100:
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
        
        hf = h5py.File(filename, 'w')
        hf.create_dataset('X_train',data=X_training_data)
        hf.create_dataset('Y_train',data=Y_training_data)
        hf.close()
        print('training data loaded')

    def save_model_as_h5(self,model_file,params):
        """
        save a models params
        """
        print('saving model paramas ...')
        if not os.path.exists('h5_files'):
            os.makedirs('h5_files')

        L = len(params)//2

        hf = h5py.File(model_file,'w')
        for key in list(params.keys()):
            hf.create_dataset(key,data=np.array(params[key]))
            hf.create_dataset(key,data=np.array(params[key]))
        hf.close()
        print('model paramas saved')

    def load_h5_model(self,model_file):
        """
        load in a models params
        """
        print('loading model paramas ...')
        hf = h5py.File(model_file,'r')
        L = len(hf.keys())//2
        params = {}
        for l in range(1,L):
            params['W'+str(l)]  = np.array(hf.get('W'+str(l)))
            params['b'+str(l)] = np.array(hf.get('b'+str(l)))
        print('model paramas loaded')

        return params

    def load_h5_training_data(self,filename):
        """
        unpacks h5 data files
        """
        print('loading training data ...')
        hf              = h5py.File(filename, 'r')
        X_training_data = np.array(hf.get('X_train'))
        Y_training_data = np.array(hf.get('Y_train'))
        key_str         = str(hf.get('Key'))
        key             = eval(key_str)
        print('training data loaded')

        return X_training_data,Y_training_data

    def resize_image(self,image,size):
        """
        resize an image
        """
        img = Image.open(image)
        img = img.resize((size,size))
        arr = np.array(img)
        
        #flatten 
        arr = arr.flatten()
        arr = arr.reshape((size*size*3),-1)
        return arr

    def show_image(self,image):
        """
        simple function to display an image
        """
        im = Image.open(image)  
        im.show()

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
        r = np.maximum(0,z)
        return r

    def initalize_params(self,layer_dims,initalization='Zero'):
        """
        initialises w,b to be zero
        INCOMPLETE
        """
        # layers dims = num of nodes in each dim, each layer in a list

        params = {}
        L = len(layer_dims)            # integer representing the number of layers
    
        if initalization == 'Zero':
            for l in range(1, L):
                params['W' + str(l)] = np.zeros((layer_dims[l],layer_dims[l-1]))
                params['b' + str(l)] = np.zeros((layer_dims[l],1))

                print('W'+str(l),params["W" + str(l)].shape)
                print('b'+str(l),params["b" + str(l)].shape)

        if initalization == 'Random':
            pass

        return params

    def update_params(self,params,grads,learning_rate,optimization='None'):
        """
        updates params
        INCOMPLETE
        """
        L = len(params) // 2

        for l in range(1,L):
            params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
            params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]


        
        # if optimization == 'None':
        #     w = w - learning_rate*dw
        #     b = b - learning_rate*db

        # if optimization == 'Adam':
        #     E     = 1e-8
        #     beta1 = 0
        #     beta2 = 0

        #     Sdw = (beta2*Sdw+(1-beta2)*dw**2)/(1-beta2)
        #     Sdb = (beta2*Sdb+(1-beta2)*db**2)/(1-beta2)
        #     Vdw = (beta1*Vdw+(1-beta1)*dw)/(1-beta1)
        #     Vdb = (beta1*Vdb+(1-beta1)*db)/(1-beta1)

        #     w = w - learning_rate*(Vdw/np.sqrt(Sdw + E))
        #     b = b - learning_rate*(Vdb/np.sqrt(Sdb + E))

        # if optimization == 'RMS':
        #     E     = 1e-8
        #     beta1 = 0
        #     beta2 = 0

        #     Sdw = (beta2*Sdw+(1-beta2)*dw**2)/(1-beta2)
        #     Sdb = (beta2*Sdb+(1-beta2)*db**2)/(1-beta2)

        #     w = w - learning_rate*Sdw
        #     b = b - learning_rate*Sdb

        # if optimization == 'Momentum':
        #     E     = 1e-8
        #     beta1 = 0
        #     beta2 = 0

        #     Vdw = (beta1*Vdw+(1-beta1)*dw)/(1-beta1)
        #     Vdb = (beta1*Vdb+(1-beta1)*db)/(1-beta1)

        #     w = w - learning_rate*Vdw
        #     b = b - learning_rate*Vdb

        return params

    def cost_function(self,Y,AL):
        """
        computes the cost
        """
        m    = Y.shape[1]
        cost = (-1/m)*(np.sum((Y*np.log(AL))+(1-Y)*np.log(1-AL)))

        return cost

    def sigmoid_rev(self,Z,dA):
        """
        INCOMPLETE
        """
        A = self.sigmoid(Z) 
        return dA * (A*(1 - A))

    def relu_rev(self,Z):
        """
        INCOMPLETE
        """
        dZ = np.heaviside(Z, 1)
        return dZ

    def lin_rev(self,params,dZ,A_pre):
        """
        INCOMPLETE
        """
        m = A_pre.shape[1]
        W = params['w']
        b = params['b']

        dw = (1/m)*np.dot(dZ,A_pre.T)
        db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        dA_pre = np.dot(W.T,dZ)

        return dA_pre,dw,db

    def forward_prop(self,X,params):
        """
        forward prop
        INCOMPLETE
        """

        Z_cache = {}
        A_cache = {}

        L = len(params) // 2

       # m = X.shape[1]        
        A = X
        for l in range(1,L):
            print('A'+str(l-1),A.shape)
            w = params['W'+str(l)]
            b = params['b'+str(l)]
            z = np.dot(w,A)+b
            A = self.relu(z)
            print('Z'+str(l),z.shape)
            Z_cache['Z'+str(l)]=z
            A_cache['A'+str(l-1)]=A


        #final activation
        wL=params['W'+str(L)]
        bL=params['b'+str(L)]

        zL  = np.dot(wL,A)+bL
        AL  = self.sigmoid(zL)
        print('AL',AL.shape)

        Z_cache['Z'+str(L)]=zL
        A_cache['A'+str(L)]=AL

        return Z_cache, A_cache

    def back_prop(self,X,Y,params,Z_cache,A_cache):
        """
        back prop
        working backwards through the nn 
        so sigmoid then relus
        """
        L = len(params)//2
        AL = A_cache['A'+str(L)]
        print('YL',Y.shape)
        #Y = Y.reshape(AL.shape)

        gradients = {}

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        ZL = Z_cache['Z'+str(L)]
        dZL = self.sigmoid_rev(ZL,dAL)
        print('dZL',dZL.shape)

        params_= {}
        params_['w'] = params['W'+str(L)]
        params_['b'] = params['b'+str(L)]

        gradients['dA'+str(L-1)],gradients['dW'+str(L)],gradients['db'+str(L)] = self.lin_rev(params_,dZL,AL)
        
        for l in reversed(range(L-1)):
            
            params_={}
            params_['w'] = params['W'+str(l+1)]
            params_['b'] = params['b'+str(l+1)]
            Z = Z_cache['Z'+str(l+1)]
            dZ = self.relu_rev(Z)#?
            print('dZ'+str(l),dZ.shape)
            A = A_cache['A'+str(l)]
            gradients['dA'+str(l)],gradients['dW'+str(l+1)],gradients['db'+str(l+1)] = self.lin_rev(params_,dZ,A)
            print('db'+str(l+1),gradients['db'+str(l+1)].shape)
            print('dW'+str(l+1),gradients['dW'+str(l+1)].shape)

        return gradients

    def gradient_descent(self,X,Y,params,epochs,learning_rate):
        """
        uses propagation in loop to optimize paramaters
        """

        costs = []
        print('x',X.shape)
    
        for i in range(epochs):
            print('\nepoch {}'.format(i+1))
            #print(params)
        
            #forward prop
            Z_cache, A_cache = self.forward_prop(X,params)

            AL = A_cache['A'+str(len(params)//2)]
            cost = self.cost_function(Y,AL) 
            #print('cost-'+str(i+1),cost)

            #back prop
            gradients = self.back_prop(X,Y,params,Z_cache,A_cache)

            # save costs values every 100 vals
            if i % 100 == 0:
                costs.append([i,cost])
        
            # update params using the learning rate
            params = self.update_params(params,gradients,learning_rate)
            
        # ax = plt.figure()
        # ax.scatter(costs[0],costs[1])
        # ax.xlabel('# of iterations')
        # ax.ylabel('Cost')
        # plt.show()
        #plt.savefig('cost.png')
        
        return params, gradients, costs

    def predict_output(self,X,params):
        """
        predicts binary label
        """
        m = X.shape[1]

        #initalise output as 0
        Y_prediction = np.zeros((1,m))

        L = len(params)//2

        A = X

        for l in range(1,L):
            w = params['W'+str(l)]
            b = params['b'+str(l)]
            z = np.dot(w,A)+b
            A = self.relu(z)

        w=params['W'+str(L)]
        b=params['b'+str(L)]

        z  = np.dot(w,A)+b
        A  = self.sigmoid(z)
        #round values to be binary 
        for i in range(A.shape[1]):

            if A[0][i] <= 0.5:
                Y_prediction[0][i] = 0

            if A[0][i] > 0.5:
                Y_prediction[0][i] = 1

        return Y_prediction

    def model(self,X_train,Y_train,layer_dims,epochs=2000,learning_rate=0.5):
        """
        will train model
        """
        print('training model ...')
        m  = X_train.shape[0]
        params = self.initalize_params(layer_dims)

        parameters,gradients,costs = self.gradient_descent(X_train,Y_train,params,epochs,learning_rate)
        
        # w = parameters["w"]
        # b = parameters["b"]

        Y_prediction_train = self.predict_output(X_train,parameters)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print('model trained')
        return parameters

    def test_image(self,image,params,image_category):
        """
        test a model with an image
        """
        pixel_dim=64
        X_training_data = np.empty((pixel_dim*pixel_dim*3,1))
        Y_training_data = np.empty((1,1))

        x_test = self.resize_image(image,pixel_dim)/255
        key    = self.get_key('dataset/bing')

        y_val  = list(key.keys())[list(key.values()).index(image_category)]
        y_test = np.array([y_val]).T

        print('#'*15)
        Y_prediction_test = self.predict_output(x_test,params)
        acc = 100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100

        print("test image predicted as {} with {} % accuracy".format(key[int(np.squeeze(Y_prediction_test))],acc))
        self.show_image(image)

###########################################
# End of Class
###########################################
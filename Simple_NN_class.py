
import numpy as np
import os

###########################################

class Simple_NN():
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
        return 0

    def train(self,num_of_layers,num_of_nodes,traing_data_dir):
        """
        INCOMPLETE
        """
        for dir in os.listdir(traing_data_dir):
            print (dir)

        return 0

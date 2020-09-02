from Simple_NN_class import *

nn = Simple_NN()
nn.format_training_data_as_h5('dataset/bing')
X_training_data,Y_training_data=nn.load_h5_training_data('training_data.h5')

w,b =nn.model(X_training_data,Y_training_data,num_iterations=2000,learning_rate=0.5)
nn.test_image('test_image.jpg',w,b)
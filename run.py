from Simple_NN_class import *

# init class instance
nn = Simple_NN()

# download training data from bing
#num_of_images_per_category = 10000
#list_of_categories         = ['Cat','Dog']
#nn.download_training_data(num_of_images_per_category,list_of_categories)

# generate .h5 formated training data file
#nn.format_training_data_as_h5('dataset/bing','h5_files/training_data.h5')

# load training data in from .h5 file
X_training_data,Y_training_data=nn.load_h5_training_data('h5_files/training_data.h5')

# generate model and save params for testing the model
w,b =nn.model(X_training_data,Y_training_data,num_iterations=2000,learning_rate=0.5)
nn.save_model_as_h5('h5_files/model_10k.h5',w,b)

# load in model params
w,b = nn.load_h5_model('h5_files/model_10k.h5')

# test model with images
nn.test_image('test_images/test_image.jpg', w,b,'Cat')
nn.test_image('test_images/test_image3.jpg',w,b,'Dog')
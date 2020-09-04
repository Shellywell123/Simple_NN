# Simple_NN
A small python 3 project written as a stepping stone for learning the basics of convolution neural networks for another repo [Pykadex](www.github.com/sudini1412/Pykadex). The project uses 10,000 images of cats and dogs downloaded from bing and uses them to train a simple nueral network. I have tried to write the class as flexible as possible so that it can be used to test any two category of images not just cats and dogs.

## Basic Method
Using simple linear regression the program optimizes the w,b paramaters by passing forwards and backwards propagration in an iterative loop. 
The program uses the sigmoid activation function to classisfy an image into one of two categories values (0 or 1) that refer to a dictionary. The w,b paramaters can then be reused to test the model with a supplied image. The program `run.py` calls function from a class I have written in `Simple_NN_class.py`.

## For running the program
I recieved some runtime warnings therfore use the flag `-W ignore`.
```bash
foo@bar:~$ python3 -W ignore run.py
```
## For downloading training data
The code in its current state does not download the training data nor is the data stored on the git, due to file size restrictions. However this can easily be done with supplied code in a few steps.\\

### 1st install bing_image_downloader v1.0.2
```bash
foo@bar:~$ pip3 install bing_image_downloader==1.0.2
```
### 2nd replace bing.py file in the package
the location of the package can be found by the command:
```bash
foo@bar:~$ pip3 show bing_image downloader
```
Open the `dir` in which the `bing_imaage_downloader` is stored in and replace the `bing.py` file with `replacement_bing1-0-2.py.` 

### 3rd Call the `download_training_data()` funtion
Then uncomment the `download_training_data()` function in `run.py.`
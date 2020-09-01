
# MOVE THIS FILE INTO A SEPERATE LOCATION ON YOUR MACHINE
from bing_image_downloader import downloader
import os

#######################################################

# put list of objects here

lim  = "10000"
pets = ["Cat","Dog"]

for pet in pets:
    downloader.download(pet, limit=lim, adult_filter_off=True, force_replace=False)

#######################################################

#summary of downloads

for pet in pets:
    print ('{} Images of {}'.format(lim,pet))


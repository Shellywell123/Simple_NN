
# MOVE THIS FILE INTO A SEPERATE LOCATION ON YOUR MACHINE
from bing_image_downloader import downloader
import os

#######################################################

# put list of objects here

lim  = "10000"
categories = ["Cat","Dog"]

for category in categories:
    downloader.download(category, limit=lim, adult_filter_off=True, force_replace=False)

#######################################################

#summary of downloads
for category in categories:
    print ('{} Images of {}'.format(lim,pet))

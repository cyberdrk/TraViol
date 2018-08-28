# TraViol 

An Automatic Traffic Violation Detection System 
 
> Statistics show: 
- 1 min we have at least 22 traffic violations in India and 
- 1 death per 4 mins due to road accidents 

This project aims to catch 

## Getting Started 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

Open the Terminal (cmd in case of Windows), navigate to your desired folder and type in the following command to download the repository: 

``` 
git clone https://github.com/cyberdrk/TraViol.git 
``` 

### Prerequisites 

The code can be run on Python 2. Please make sure you have at least Python 2.7.x running on your machine along with the latest releases of the following packages. 

```
*Software: 
Python 2.7.x 

*Packages: 
cv2, numpy, sys, glob, PIL.Image, pytesseract, argparse, math, time, os and matplotlib 
```

### Installing 

Clone or download the repository. 

Open the respective python files in your favorite editor. 

Everything is pretty much there. Go ahead and run them! 


## Running 

* [km.py](/km.py) : This script flags off capturing the video stream. It performs background segmentation on the video using the Mixture of Gaussian method and some morphological operations to detect cars. Blobs of cars are detected using the cv2.moments method and cv2.contours method. 

* [trorb.py](/trorb.py) : These files contain the main Overspeeding script. 

* [pyANPD.py](/pyANPD.py) : If a violation is detected, this script extracts the ROI, i.e., the number plate from the frame containing the car 

* [tesse.py](/tesse.py). This script reads the chracters on the number plate (from the ROI extracted using the above script) using the Tesseract library. 

## PowerPoint Presentation for the Project : https://docs [dot] google [dot] com/presentation/d/17EBQIEI1Dg7hZalhWzVDqoJervvtFXIorkPj9AHB3Hk/edit?usp=sharing

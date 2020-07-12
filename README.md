# Introduction
 This repo is a part of a medical robot that serves in hospital or clinic. The robot can check human temperature and give out (free) hand sanitizer and (free) mask if requested by comunicating with an external Adruino board.
 It can also regconize patients if they've signed up and check if there were a mask on thier face. This is done by using 2 models separately. 
 for the GUI, tkinter was used for its simplicity.
 
 ## How to use 
 - upload ``` mask.ino ``` file to an Arduino board
 - Stick it into your laptop
 - run ```GUI.py``` file and you're good to go   
 [Disclaimer]: Without an Arduino borad connected, the fps could be fluctuated as the programe will try to read temp from sensor repeatedly.  
 The programe is written to run in Linux.
 
 ## Prerequisites
 
 - tensorflow 2.x
 - keras 
 - dlib 19.x +
 - keras_vggface
 - glob
 - pickle
 - serial
 - and openCV of course
 - adruino
 - mlx90614 - infrared thermometer
 - a proximity sensor (any type of it, in this case, an ir sensor was used as it cost just nearly 0.5 buck

## Running the test
### On the Arduino
Connect sensors to arduino as the image below
![](data/examples/diagram.jpg)  
If anything gets close to the proximity sensor, adruino will read the temp from mlx90614. As the signal from thermometer is analog, there would be no lag or delay. The signal wlll then be processed by ``` <Adafruit_MLX90614.h>``` library, what we receive is just a very fine processed temperature. However, each sensor will need to be added an offset value (ranging to 3.5 Celsius). In this case, the value is initually set by 1.4.  
Next step, upload ``` mask.ino ``` file to an Arduino board and plug it to computer
### On the computer
The GUI will have an apperance like this  
![](data/examples/init.png)  
if this is the first time the computer've seen new user then spend a littie time to sign up. An window will pop up to collect user's images
![](data/examples/sign_up.png) 
wait a few seconds (actually it takes "some" few seconds to complete adding new user)  
![](data/examples/done.png)  
Alright, from now on, robot should have a new friend :v.  
### how about face mask and hand sanitizer, you said it gives freely???
- For the face mask, press ``` Have mask``` button and a white box will slide to you.
- If you want to clean your hand, put it on the hole beside white box.
- However in the scope of this repo, how they works won't be included. 
![](data/examples/face_mask_2.jpeg)  
Sorry, the robot looks too ugly, it lives on budget :(
- Don't foget to check your temp by moving your forehead to this little stick.
![](data/examples/temp.jpg)
- Then, your information will be saved in ```data/xlsx/yyyy-mm-dd.xlsx```

# What are the models in this repo ???

This robot has 2 models on his brain. 1 is to recognize users, the other is to check if they are putting on a mask. Both of these two used Transfer Learning method
## Face recognition
With the Backbone is VGGFace2, this model gives an incredible result (with non-asian face). Without my face signed up, it confused me between strangers and Obama's First lady - Michelle Obama (?????). Can't bilieve we have any kind of these blood ties.
The thing is, this model is built for non-asian faces and it needs some change in its weight.
### Data for tuning


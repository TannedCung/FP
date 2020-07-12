# Introduction
 This repo is a part of a medical robot that serves in hospital or clinic. The robot can check human temperature and give out mask if requested by comunicating with an external Adruino board.
 It can also regconize patients if they've signed up and check if there were a mask on thier face. This is done by using 2 models separately. 
 for the GUI, tkinter was used for its simplicity.
 
 ## How to use 
 Simply run GUI.py file   
 [Disclaimer]: Without an Arduino port connected, the fps could be fluctuated as the programe will try to read temp from sensor repeatedly.  
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

## Running the test

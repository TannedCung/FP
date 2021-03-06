import serial
import numpy as np


class Serial_stuff():
    """
     Connect to the arduino
    """

    def __init__(self):
        self.ser = None
        for i in range(5):
            port = "/dev/ttyACM{}".format(i)
            try:
                self.ser = serial.Serial(port,baudrate=9600)
            except Exception as e:
                print("[ERROR]: {}".format(e))
            if self.ser != None:
                print("[INFO]: Connected to /dev/ttyACM{}".format(i))
                break
        try:
            self.ser.open()
        except Exception as e:
            print ("[ERROR]: Opening serial port: " + str(e))


    def get_temp(self):
            # print(temper)
        try:
            temper = self.ser.readline()
            temp = float(str(temper)[2:4]) + float(str(temper)[5:7])/100
            return temp
        except Exception as e:
            print("[ERROR]: {}".format(e))
            temp = 0
        # return 37.5
        
        
    def out(self):
        self.ser.write(bytes("out", encoding="UTF-8"))                                        # "out" = give out function in adruino

    def stop(self):
        self.ser.write(bytes("stop", encoding="UTF-8"))                                       # need to concanate 3 command together for faster speed
                                                                            # "stop" = stop function that make the robot stop running in arduino
    def run(self):
        self.ser.write(bytes("run", encoding="UTF-8"))                                        # "run" = continue running/ start running in arduino
                                                                            # "stop" = stop function that make the robot stop running in arduino
                                                                            
"""
a = Serial_stuff()

while 1:
    t = a.get_temp()
    print(t)
"""
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import threading
import compute_feature as compute_feature
import Student
import Recognition2 as Recognition
import cv2
import numpy as np
import dlib
from keras_vggface import VGGFace
from Transporter import *
import sys

class Reception():
    Cascade_path = "./pretrained_models/haarcascade_frontalface_alt.xml"

    def __init__(self):          
        self.Receptionist = compute_feature.FaceExtractor()
        self.the_bodyguard = Recognition.FaceIdentify()
        self.new_std = Student.Student()
        self.vs = cv2.VideoCapture(-1)
        self.detector = cv2.CascadeClassifier(self.Cascade_path)
        self.Jas = Serial_stuff()
        self.t = None
        self.exit_command = False

        # init GUI parameters
        self.root = Tk()
        self.root.title("Hi there!")
        self.stand_by = {}
        for i in range(4):
            self.stand_by[i] = ImageTk.PhotoImage(Image.open('sb_{}.png'.format(i+1)))
            
        self.bt_home = ImageTk.PhotoImage(Image.open('home.png'))
        self.bt_have_mask = ImageTk.PhotoImage(Image.open('have_mask.png'))
        self.bt_sign_up = ImageTk.PhotoImage(Image.open('sign_up.png'))
        self.bt_quit = ImageTk.PhotoImage(Image.open('quit.png'))
        self.bt_sign_up_2 = ImageTk.PhotoImage(Image.open('sign_up_2.png'))


        self.panel = None                   # for the video feed
        self.in4 = {}                       # for the infomation of users
        self.img_in4 = {}                   # for the imgs of users

        HEIGHT = 550
        WIDTH = 1225

        self.canvas = Canvas(self.root, height=HEIGHT, width=WIDTH)
        self.canvas.pack()
        self.back_ground_img = PhotoImage(file='bg.png')
        self.back_ground = Label(self.root, image=self.back_ground_img)
        self.back_ground.place(relheight=1, relwidth=1 )
        self.grid1 = Frame(self.root, bg='#1a8cff', bd=5)
        self.grid1.place(relx=0.05, rely=0.05)
        self.grid2 = Frame(self.root, bg='#1a8cff', bd=5, width=0.3*WIDTH)
        self.grid2.place(relx=0.63, rely=0.05, height=490)

        self.option_frame = Frame(self.grid2, bg='#99bbff', bd=0)
        self.option_frame.place(relx=0, rely=0, height=35, relwidth=1)
        self.info_frame = Frame(self.grid2, bg='#99bbff', bd=1)
        self.info_frame.place(relx=0, y=40, height=440, relwidth=1)
        
        # mode selection
        self.options = ['Normal', 'Sign in']
        self.Home = Button(self.option_frame, bd=0, image=self.bt_home, command=self.Normal)
        self.Home.place(relx=0, rely=-0.02)
        self.sign_up = Button(self.option_frame,bd=0, image=self.bt_sign_up, command=self.signIn)
        self.sign_up.place(relx=0.25, rely=-0.02)
        self.quit = Button(self.option_frame, bd=0, image=self.bt_have_mask, command=self.mask_out)
        self.quit.place(relx=0.5, rely=-0.02)
        self.quit = Button(self.option_frame, bd=0, image=self.bt_quit, command=self.Exit)
        self.quit.place(relx=0.75, rely=-0.02)

        self.sign_in_frame = Frame(self.info_frame, bg='#99bbff')
        self.sign_in_frame.grid(column=0, row=1)  
        self.normal_frame = Frame(self.info_frame, bg='#99bbff' )
        self.normal_frame.grid(column=0, row=1)

        color = ['#ffbf80', '#bfff80', '#9fff80', '#80ffdf']
        for i in range(4):
            self.in4[i] = Label(self.normal_frame, text="None", 
                    font=('VnCooper', 11, 'bold'), bg=color[i], height=7, width=27, justify="left")
            self.in4[i].grid(column=1, row=i, sticky="e")
            self.img_in4[i] = Label(self.normal_frame, image=self.stand_by[i], bg='#ff9999', bd=3)
            self.img_in4[i].grid(column=0, row=i)

        self.thread = threading.Thread(target=self.VS)
        self.thread.daemon = True
        self.temp_thread = threading.Thread(target=self.Jason)
        self.temp_thread.daemon = True
        # self.thread2 = threading.Thread(target=self.signIn)

        self.temp_thread.start()
        self.thread.start()
        sign_in_mode = False

        
    def mode(self, event):
        if self.dropDown.get() == self.options[0]:
            self.Normal()
        elif self.dropDown.get() == self.options[1]:
            self.signIn()

    def forget(self, widget):
        widget.grid_forget()

    def retrieve(self, widget, col=0, row=1): 
        widget.grid(column=col, row=row)
    
    def clear_name(self, event):
        self.name.delete(0, END) 

    def clear_ID(self, event):
        self.id.delete(0, END)

    def Exit(self):
        self.exit_command = True
        self.root.destroy()
        # self.temp_thread.join()
        # cself.thread.exit(0)
        # self.thread.join()
        sys.exit(0)
        self.vs.release()

    def Normal(self):
        # delete sign in frame
        self.forget(self.sign_in_frame)
        # show normal frame
        self.retrieve(self.normal_frame)



    def signIn(self):
        # delete normal options
        self.forget(self.normal_frame)
        # show sign in frame
        self.retrieve(self.sign_in_frame)

        global sign_in_mode, thread
        sign_in_mode = True
        self.name = Entry(self.sign_in_frame, width=38, font=('courier 10 pitch', 11, 'bold'))
        self.name.insert(0, "Your full name: ")
        self.name.bind("<Button-1>", self.clear_name)
        self.name.grid(column=0, row=0)
        
        self.id = Entry(self.sign_in_frame, width=38, font=('courier 10 pitch', 11, 'bold'))
        self.id.insert(0, "ID: ")
        self.id.bind("<Button-1>", self.clear_ID)
        self.id.grid(column=0, row=1)
        

        sub = Button(self.sign_in_frame, text="Submit", bd=0, image=self.bt_sign_up_2, command=self.submit)
        sub.grid(column=0, row=2)


    def submit(self):
        global sign_in_mode
        n = str(self.name.get())
        i = int(self.id.get())
        sy = i//10000 - 1955
        self.new_std.save_infor(name=n, id=i, school_year=sy)
        # self.dropDown.set(self.options[0])
        self.Receptionist.extract_faces(name=n,cap=self.vs)
        # thread2.join()
        self.Receptionist.compute_features(name=n)
        self.Receptionist.biuld_new_model()
        self.the_bodyguard.reload_model()
        self.reset_tracker()
        sign_in_mode = False


    def remove_bad_tracker(self, frame):
        delete_id_list = []

        # Thru all the detected faces
        for faceID in self.faceTracker.keys():
            trackedPosition = self.faceTracker[faceID].get_position()
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            # with tracking conf < 4, that shouldn't be an actual face
            if self.faceTracker[faceID].update(frame) < 4 or t_x<=0 or t_y<=0:
                delete_id_list.append(faceID)

        # pop the bad face
        for faceID in delete_id_list:
            self.faceTracker.pop(faceID, None)
            self.faceCurrentPos.pop(faceID, None)
            self.faceStatus.pop(faceID, None)
            self.label.pop(faceID, None)
    
    def reset_tracker(self):
        self.faceTracker = {}
        self.faceCurrentPos = {}
        self.faceStatus = {} # 0,1,2,3,4,5,6,7,8,9,10 = "Detecting"
                        # 11 = "Unknown"
                        # 12 = "Known"
        self.label = {}  # contains name
        self.score = {}
        self.temp = {}  # a list of each one's temperature

    def show_info(self, imgs, info, quantity, listID):
        # find the biggest face, that face should be the one we are measuring 
        t_w = 0
        best = None

        for id in listID:
            size = self.faceTracker[listID[id]].get_position()
            if t_w < int(size.width()):
                t_w = int(size.width())
                best = listID[id]
        
        if self.t != None:
            self.temp[best] = self.t
            self.t = None

        # forget everone disapeared 
        for i in range(4-quantity):
            self.in4[3-i].configure(text="None")
            self.in4[3-i].text="None"
            self.img_in4[3-i].configure(image=self.stand_by[3-i])
            self.img_in4[3-i].image = self.stand_by[3-i]


        for i in range(quantity):
            img = imgs[i]
            i4 = info[i]
            n = info[i].get('name')
            ID = info[i].get('ID')
            school_year = info[i].get('school_year')
            temp = self.temp[listID[i]]


            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.in4[i].configure(text=" {} \n {} \n K{} \n {} C".format(n, ID, school_year, temp))
            self.img_in4[i].configure(image=img)
            self.img_in4[i].image = img

    def is_blurry(self, image, threshold=150):
        res = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if res > threshold:
            return 0
        else:
            return 1

    def mask_out(self):
        self.Jas.out()


    def Jason(self):                                
        """ Jason can talk with the arduino """
        while 1:
            self.t = self.Jas.get_temp()

    
    def VS(self):
        thread = threading.current_thread()
        frame_counter = 0
        face_number = 0

        self.faceTracker = {}
        self.faceCurrentPos = {}
        self.faceStatus = {} # 0,1,2,3,4,5,6,7,8,9,10 = "Detecting"
                        # 11 = "Unknown"
                        # 12 = "Known"
        self.label = {}  # contains name
        self.score = {}
        self.temp = {}   # a list of each one's temperature
        face_pics = np.empty((4, 100, 100, 3), np.uint8) # placeholder for the side images
        info = {}
        listID = {}

        # infinite loop, break by key ESC
        while self.exit_command == False :
            _, frame = self.vs.read()
            frame = np.fliplr(frame)
            frame = np.array(frame)
            frame_counter += 1

            self.show_info(imgs=face_pics, info=info, quantity=len(info), listID=listID)
            info = {}
            listID = {}

            self.remove_bad_tracker(frame=frame)
            if not (frame_counter % 10):
                # draw info in the panel
                face_imgs, points = self.the_bodyguard.detect_face(frame)
                # thru detected faces
                for (_x, _y, _w, _h) in points:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    # cal center of the face
                    x_center = x + 0.5 * w
                    y_center = y + 0.5 * h  
                    matchFaceID = None
                    # thru tracked faces, check if this i-th face were a new face
                    for faceID in self.faceTracker.keys():
                        trackedPosition = self.faceTracker[faceID].get_position()

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())   

                        t_x_center = t_x + 0.5 * t_w
                        t_y_center = t_y + 0.5 * t_h
                        # if the center is out of the rects of existing faces, then it is no longer an existing face   
                        if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
                            matchFaceID = faceID

                    if matchFaceID is None:
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(frame, dlib.rectangle(x, y, x+w, y+h))

                        self.faceTracker[face_number] = tracker
                        self.faceStatus[face_number] = 0
                        self.temp[face_number] = None
                        face_number += 1
            acquantance_count = 0
            for faceID in self.faceTracker.keys():
                trackedPosition = self.faceTracker[faceID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                self.faceCurrentPos[faceID] = (t_x, t_y, t_w, t_h)
                # detect mask and draw the result
                mask = self.the_bodyguard.is_mask_on(frame=frame, face_area=self.faceCurrentPos[faceID])
                self.the_bodyguard.draw_mask_stt(frame=frame, point=self.faceCurrentPos[faceID], state=mask)
                
                # faceTracker[faceID].update(frame)
                # cv2.rectangle(frame, (t_x,t_y), (t_x+t_w, t_y+t_h), (225,225,0))
                # if the face state (status) is Unknown, draw red rect,
                # no more recognition
                if self.faceStatus[faceID] == 11: # a stranger
                    frame = self.the_bodyguard.draw_label(frame=frame, point=self.faceCurrentPos[faceID],
                                            score=self.score[faceID], flag=self.faceStatus[faceID], label=self.label[faceID])
                # if the face state is Known, draw that person information
                elif self.faceStatus[faceID] >= 12: # an acquantance
                    if acquantance_count<=3 and t_x>0 and t_y>0:
                        save = frame[t_y:t_y+t_h, t_x:t_x+t_w]
                        face_pics[acquantance_count] = cv2.resize(save, (100,100))
                        info[acquantance_count] = self.new_std.search_info(name=self.label[faceID])
                        listID[acquantance_count] = faceID
                        acquantance_count = acquantance_count + 1
                        

                    frame = self.the_bodyguard.draw_label(frame=frame, point=self.faceCurrentPos[faceID],
                                            score=self.score[faceID], flag=self.faceStatus[faceID], label=self.label[faceID])
                # if this is the 1st time detected, recognize it
                elif self.faceStatus[faceID] == 0:
                    face, pos = self.the_bodyguard.crop_face(frame, self. faceCurrentPos[faceID])
                    # face = np.expand_dims(face, axis=0)
                    # name, score = self.the_bodyguard.identify_face(face)
                    self.score[faceID] = 0
                    self.label[faceID] = "Unknown"
                    frame = self.the_bodyguard.draw_label(frame=frame, point=self.faceCurrentPos[faceID],
                                            score=self.score[faceID], flag=0 )
                    self.faceStatus[faceID] += 1
                # in 10 times of recognizing if at any time model relize an acquantian
                # name that acquantain
                elif self.faceStatus[faceID] < 11 and self.faceStatus[faceID] > 0:
                    face, pos = self.the_bodyguard.crop_face(frame, self.faceCurrentPos[faceID])
                    # only identify the vivid face, avoid blurred face makes "Unknown" result
                    if not self.is_blurry(face):
                        face = np.expand_dims(face, axis=0) # shape = (1,224,224,3)
                        name, score = self.the_bodyguard.identify_face(face)
                        if name != "Unknown":
                            self.label[faceID] = name
                            self.score[faceID] = score
                            self.faceStatus[faceID] = 12
                        else:
                            self.faceStatus[faceID] += 1

                    frame = self.the_bodyguard.draw_label(frame=frame, point=self.faceCurrentPos[faceID],
                                                score=self.score[faceID], flag=self.faceStatus[faceID] if self.faceStatus[faceID] <12 else 11)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            if self.panel is None:
                self.panel = Label(self.grid1, image=image)
                self.panel.image = image
                self.panel.grid(column=0, row=0)
            else:
                self.panel.configure(image=image)
                self.panel.image = image

            if cv2.waitKey(5) == 27 or self.exit_command == True:
                break

rct = Reception()
rct.root.mainloop()

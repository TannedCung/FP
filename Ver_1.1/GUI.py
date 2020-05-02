from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import threading
import compute_feature_v2 as compute_feature
import Student
import Recognition2_v2 as Recognition
import cv2
import numpy as np
import dlib
from keras_vggface import VGGFace

class Reception():

    def __init__(self):          
        self.Receptionist = compute_feature.FaceExtractor()
        self.the_bodyguard = Recognition.FaceIdentify()
        self.new_std = Student.Student()
        self.vs = cv2.VideoCapture(0)

        # init GUI parameters
        self.root = Tk()
        self.root.title("Hi there!")
        self.stand_by = ImageTk.PhotoImage(Image.open('stand_by.png'))
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
        self.grid2.place(relx=0.63, rely=0.05, relheight=0.9)

        self.option_frame = Frame(self.grid2, bg='#99bbff', bd=5)
        self.option_frame.place(relx=0, rely=0, relheight=0.1, relwidth=1)
        self.info_frame = Frame(self.grid2, bg='#99bbff', bd=5)
        self.info_frame.place(relx=0, rely=0.11, relheight=0.89, relwidth=1)
        
        # mode selection
        self.options = ['Normal', 'Sign in']
        self.dropDown = ttk.Combobox(self.option_frame, value=self.options)
        self.dropDown.current(0)
        self.dropDown.bind("<<ComboboxSelected>>", self.mode)
        self.dropDown.place(relx=0.2, rely=0, relwidth=0.6)

        self.sign_in_frame = Frame(self.info_frame, bg='#99bbff')
        self.sign_in_frame.grid(column=0, row=1)  
        self.normal_frame = Frame(self.info_frame, bg='#99bbff' )
        self.normal_frame.grid(column=0, row=1)

        color = ['#ffbf80', '#bfff80', '#9fff80', '#80ffdf']
        for i in range(4):
            self.in4[i] = Label(self.normal_frame, text="None", 
                    font=('VnCooper', 11, 'bold'), bg=color[i], height=7, width=27, justify="left")
            self.in4[i].grid(column=1, row=i, sticky="e")
            self.img_in4[i] = Label(self.normal_frame, image=self.stand_by, bg='#ff9999', bd=3)
            self.img_in4[i].grid(column=0, row=i)

        self.thread = threading.Thread(target=self.VS)
        self.thread2 = threading.Thread(target=self.signIn)
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

    def Normal(self):
        # delete sign in frame
        self.forget(self.sign_in_frame)
        # show normal frame
        self.retrieve(self.normal_frame)
        print('normal')



    def signIn(self):
        # delete normal options
        self.forget(self.normal_frame)
        # show sign in frame
        self.retrieve(self.sign_in_frame)

        global sign_in_mode, thread
        sign_in_mode = True
        
        self.name = Entry(self.sign_in_frame, width=50, font=('VnCooper', 11, 'bold'))
        self.name.grid(column=0, row=0)
        self.name.insert(0, "Tell me your name: ")
        self.id = Entry(self.sign_in_frame, width=50, font=('VnCooper', 11, 'bold'))
        self.id.grid(column=0, row=1)
        self.id.insert(0, "Your ID: ")
        self.school_year = Entry(self.sign_in_frame, width=50, font=('VnCooper', 11, 'bold'))
        self.school_year.grid(column=0, row=2)
        self.school_year.insert(0, "And your schoolyear: ")

        sub = Button(self.sign_in_frame, text="Submit", command=self.submit)
        sub.grid(column=0, row=3)


    def submit(self):
        global sign_in_mode
        if not sign_in_mode: 
            thread2.start()
        n = str(self.name.get())
        i = int(self.id.get())
        sy = int(self.school_year.get())
        self.new_std.save_infor(name=n, id=i, school_year=sy)
        # self.dropDown.set(self.options[0])
        self.Receptionist.extract_faces(name=n,cap=self.vs)
        # thread2.join()
        self.Receptionist.compute_features(name=n)
        self.the_bodyguard.reload_feature_map()
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
        self.faceStatus = {} # 0,1,2,3,4 = "Detecting"
                        # 5 = "Known"
                        # 6 = "Unknown"
        self.label = {}  # contains name


    def show_info(self, imgs, info, quantity):

        # forget everone disapeared 
        for i in range(4-quantity):
            self.in4[3-i].configure(text="None")
            self.in4[3-i].text="None"
            self.img_in4[3-i].configure(image=self.stand_by)
            self.img_in4[3-i].image = self.stand_by


        for i in range(quantity):
            img = imgs[i] if len(imgs) > 1 else imgs
            n = info[i].get('name')
            ID = info[i].get('ID')
            school_year = info[i].get('school_year')
            temp = info[i].get('temperature')

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.in4[i].configure(text=" {} \n {} \n K{} \n {} C".format(n, ID, school_year, temp))
            self.img_in4[i].configure(image=img)
            self.img_in4[i].image = img
            

    
    def VS(self):
        thread = threading.current_thread()
        frame_counter = 0
        face_number = 0

        self.faceTracker = {}
        self.faceCurrentPos = {}
        self.faceStatus = {} # 0,1,2,3,4 = "Detecting"
                        # 5 = "Known"
                        # 6 = "Unknown"
        self.label = {}  # contains name
        face_pics = np.empty((4, 100, 100, 3), np.uint8)
        info = {}

        # infinite loop, break by key ESC
        while True :
            _, frame = self.vs.read()
            frame = np.fliplr(frame)
            frame = np.array(frame)
            frame_counter += 1
            self.remove_bad_tracker(frame=frame)
            if not (frame_counter % 10):
                # draw info to the panel
                self.show_info(imgs=face_pics, info=info, quantity=len(info))
                info = {}
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
                        face_number += 1

            for i, faceID in enumerate(self.faceTracker.keys()):
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
                if self.faceStatus[faceID] == 6:
                    frame = self.the_bodyguard.draw_label(frame=frame, point=self.faceCurrentPos[faceID],
                                            flag=6)
                # if the face state is Known, draw that person information
                elif self.faceStatus[faceID] == 5: # an acquantance
                    if i<=3 and t_x>0 and t_y>0:
                        save = frame[t_y:t_y+t_h, t_x:t_x+t_w]
                        face_pics[i] = cv2.resize(save, (100,100))
                        info[i] = self.new_std.search_info(name=self.label[faceID]) 

                    frame = self.the_bodyguard.draw_label(frame=frame, point=self.faceCurrentPos[faceID],
                                            flag=5, label=self.label[faceID])
                # if this is the 1st time detected, recognize it
                elif self.faceStatus[faceID] == 0:
                    face, pos = self.the_bodyguard.crop_face(frame,self. faceCurrentPos[faceID])
                    face = np.expand_dims(face, axis=0)
                    name = self.the_bodyguard.identify_face(face)
                    self.label[faceID] = name
                    frame = self.the_bodyguard.draw_label(frame=frame, point=self.faceCurrentPos[faceID],
                                            flag=0 )
                    self.faceStatus[faceID] += 1
                # in 5 times of recognizing that face, if at any time, the label changes,
                # that person should be a stranger
                elif self.faceStatus[faceID] < 5 and self.faceStatus[faceID] > 0:
                    frame = self.the_bodyguard.draw_label(frame=frame, point=self.faceCurrentPos[faceID],
                                                flag=self.faceStatus[faceID] )
                    if self.label[faceID] == "Unknown":
                        self.faceStatus[faceID] = 6
                    if not (frame_counter % 2):
                        face, pos = self.the_bodyguard.crop_face(frame, self.faceCurrentPos[faceID])
                        face = np.expand_dims(face, axis=0)
                        name = self.the_bodyguard.identify_face(face)
                        if self.label[faceID] == name:
                            self.faceStatus[faceID] += 1
                        elif self.label[faceID] != name:
                            self.label[faceID] = "Unknown"
                            self.faceStatus[faceID] = 6

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

            if cv2.waitKey(5) == 27:
                break

rct = Reception()
rct.root.mainloop()

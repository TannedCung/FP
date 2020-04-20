import cv2 
import numpy as numpy
import pickle
import os

def save_pikle(address, pickleFile):
    file_to_save = open(address, "wb")
    pickle.dump(pickleFile, file_to_save)
    file_to_save.close()

def load_pickle(address):
    if not os.path.exists(address):
        save_pikle(address, {})
        print ("1st init")
    file_to_load = open(address, "rb")
    pickleFile = pickle.load(file_to_load)
    file_to_load.close()
    return pickleFile

class Student:
    def __init__(self):
        self.name = 'Unknown'
        self.id = 00000000
        self.school_year = 61
        self.temp = 38

    def save_infor(self, name = 'Unknown', id = 00000000, school_year = 61):
        self.name = name
        self.id = int(id)
        self.school_year = int(school_year)

        files = list(load_pickle("./data/pickle/Students.pickle"))
        files.append(self.get_infor())
        save_pikle("./data/pickle/Students.pickle", files)

    def get_infor(self):
        return {"name": self.name,
                    "ID": self.id,
                    "school_year": self.school_year,
                    "temperature": self.temp}
'''
def main():
    Tanned = Student()
    Tanned.save_infor()


if __name__ == "__main__":
    main()
'''
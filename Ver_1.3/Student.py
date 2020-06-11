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
        # self.name = 'Unknown'
        self.id = 00000000
        self.school_year = 61
        self.data = list(load_pickle("./data/pickle/Students.pickle"))

    def save_infor(self, name, id = 00000000, school_year = 61):
        self.name = name
        self.id = int(id)
        self.school_year = int(school_year)

        self.data.append(self.get_infor())
        save_pikle("./data/pickle/Students.pickle", self.data)
        self.data = list(load_pickle("./data/pickle/Students.pickle"))

    def get_infor(self):
        return {"name": self.name,
                    "ID": self.id,
                    "school_year": self.school_year}

    def search_info(self, name):
        infor = next(item for item in self.data if item["name"] == name)
        '''
        ID = infor.get('ID')
        school_year = infor.get('school_year')
        temp = infor.get('temperature')
        return name, ID, school_year, temp
        '''
        return infor

'''
def main():
    info = list
    Tanned = Student()
    n, i, s, t = Tanned.search_info(name='barack')
    info[0] = n,i,s,t
    n, i, s, t = Tanned.search_info(name='Nguyen Phu Trong')
    info[1] = n,i,s,t
    print (info)
    print(type(info))


if __name__ == "__main__":
    main()
'''

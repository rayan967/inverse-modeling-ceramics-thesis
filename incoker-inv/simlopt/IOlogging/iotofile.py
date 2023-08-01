import os.path


class IOToLog:

    def __init__(self, filepath, filename) -> None:
        if os.path.isdir(filepath):
            jointpath = os.path.join(filepath, filename)
            try:
                self.f = open(jointpath+".txt", "a")
            except IOError:
                print("Error: File does not appear to exist.")
                return 0
            #print("Set filename and filepath")

    def closeOutputLog(self):
        self.f.close()

    def addToFile(self,  *stringToWrite) -> None:
        for string in stringToWrite:
            self.f.write(string)
        self.f.write("\n")    

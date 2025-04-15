from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pyaes, pbkdf2, binascii, os, secrets
import base64
from tkinter import ttk
from tkinter import filedialog
import os

import json
from hashlib import sha256
import socket

main = Tk()
main.title("Cross-VM Network Channel Attacks and Countermeasures within Cloud Computing Environments")
main.geometry("1300x1200")

global files, tf1

def getKey(): #generating key with PBKDF2 for AES
    password = "s3cr3t*c0d3"
    passwordSalt = '76895'
    key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
    return key

def encrypt(plaintext): #AES data encryption
    aes = pyaes.AESModeOfOperationCTR(getKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    ciphertext = aes.encrypt(plaintext)
    return ciphertext

def decrypt(enc): #AES data decryption
    aes = pyaes.AESModeOfOperationCTR(getKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    decrypted = aes.decrypt(enc)
    return decrypted

def uploadFile():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir=".")
    with open(filename, 'rb') as file:
        data = file.read()
    file.close()
    name = os.path.basename(filename)
    data = encrypt(data)
    print(len(data))
    hashcode = sha256(data).hexdigest()
    data = base64.b64encode(data)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect(('localhost', 3333))
    jsondata = json.dumps({"request": 'upload', "filename": name, "filedata": data.decode(), "hash": hashcode})
    client.send(jsondata.encode())
    data = client.recv(100)
    data = data.decode()
    text.insert(END,"Server Response : "+data+"\n\n")

def downloadFile():
    text.delete('1.0', END)
    name = tf1.get()
    jsondata = json.dumps({"request": 'download', "filename": name})
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect(('localhost', 2222))
    client.send(jsondata.encode())
    data = client.recv(100000)
    decrypted = decrypt(data)
    with open('Received/'+name, 'wb') as file:
       file.write(decrypted)
    file.close()
    text.insert(END,"File saved as "+name+" inside Received folder\n")

def getFiles():
    global files
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect(('localhost', 2222))
    jsondata = json.dumps({"request": 'files'})
    client.send(jsondata.encode())
    data = client.recv(1000)
    data = data.decode()
    data = data.strip()
    files = data.split(",")
    tf1['values'] = files
    if len(files) > 0:
        tf1.current(0)

def graph():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect(('localhost', 3333))
    jsondata = json.dumps({"request": 'graph'})
    message = client.send(jsondata.encode())
    data = client.recv(100)
    data = data.decode()
    data = data.strip()
    data = data.split(" ")
    height = [int(data[0]), int(data[1]), int(data[2])]
    bars = ('Total Request', 'Attack Request', 'Normal Request')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Attack Simulation Graph")
    plt.show()    

def close():
    main.destroy()

def runGUI():
    global text, tf1, files

    font = ('times', 15, 'bold')
    title = Label(main, text='Cross-VM Network Channel Attacks and Countermeasures within Cloud Computing Environments')
    title.config(bg='mint cream', fg='olive drab')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 14, 'bold')
    ff = ('times', 12, 'bold')

    l1 = Label(main, text='Available Files')
    l1.config(font=font1)
    l1.place(x=50,y=100)
    files = []
    files.append("Available Files")
    tf1 = ttk.Combobox(main,values=files,postcommand=lambda: tf1.configure(values=files))
    tf1.place(x=200,y=100)
    tf1.config(font=font1)
    
    uploadButton = Button(main, text="Send File Request to Cloud", command=uploadFile)
    uploadButton.place(x=50,y=150)
    uploadButton.config(font=ff)

    uploadButton = Button(main, text="Get Files From Cloud", command=getFiles)
    uploadButton.place(x=280,y=150)
    uploadButton.config(font=ff)

    downloadButton = Button(main, text="Download File", command=downloadFile)
    downloadButton.place(x=480,y=150)
    downloadButton.config(font=ff)

    graphButton = Button(main, text="Attack Simulation Graph", command=graph)
    graphButton.place(x=650,y=150)
    graphButton.config(font=ff)

    closeButton = Button(main, text="Exit", command=close)
    closeButton.place(x=50,y=200)
    closeButton.config(font=ff)

    font1 = ('times', 13, 'bold')
    text=Text(main,height=22,width=100)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=250)
    text.config(font=font1)

    main.config(bg='gainsboro')
    main.mainloop()

if __name__ == '__main__':
    runGUI()

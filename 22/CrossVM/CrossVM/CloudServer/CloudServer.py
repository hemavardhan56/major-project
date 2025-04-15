import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import json
import base64
import os
import zlib
import sys

running = True

def startCloudServer():
    class UpdateModel(Thread): 
 
        def __init__(self,ip,port): 
            Thread.__init__(self) 
            self.ip = ip 
            self.port = port 
            print('Request received from Client IP : '+ip+' with port no : '+str(port)+"\n") 
 
        def run(self): 
            data = conn.recv(1000000)
            try:
                data = json.loads(data.decode())
                request_type = str(data.get("request"))
                if request_type == 'download':
                    filename = str(data.get("filename"))
                    with open('Files/'+filename, 'rb') as file:
                        data = file.read()
                    file.close()
                    decompressed_data = zlib.decompress(data)
                    conn.send(decompressed_data)
                    print("Requested File "+filename+" Data Sent to client")
                if request_type == 'files':
                    names = ''
                    for root, dirs, directory in os.walk('Files'):
                        for j in range(len(directory)):
                            names += directory[j]+","
                    names = names.strip()
                    names = names[0:len(names)-1]
                    conn.send(names.encode())
                    print("File list sent to client")
                if request_type == 'upload':
                    filename = str(data.get("filename"))
                    filedata = data.get("filedata")
                    original_size = sys.getsizeof(filedata)
                    filedata = filedata.encode()
                    filedata = base64.b64decode(filedata)
                    print("File Saved Request Arrived : "+filename)
                    compressed_data = zlib.compress(filedata)
                    with open('Files/'+filename, 'wb') as file:
                        file.write(compressed_data)
                    file.close()          
                    conn.send(str('File saved at cloud server successfully').encode())
            except Exception:
                conn.send("Unable to process uploaded file".encode())
                pass
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('localhost', 2222))
    print("Cloud Server Started & waiting for incoming connections\n\n")
    while running:
        server.listen(4)
        (conn, (ip,port)) = server.accept()
        newthread = UpdateModel(ip,port) 
        newthread.start() 
    
def startServer():
    Thread(target=startCloudServer).start()

startServer()


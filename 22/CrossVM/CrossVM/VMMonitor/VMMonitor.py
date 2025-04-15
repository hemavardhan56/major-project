import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import json
import base64
import os
from hashlib import sha256
import socket

running = True
attack_count = 0
normal = 0
total = 0

def startMonitorServer():
    class UpdateModel(Thread): 
 
        def __init__(self,ip,port): 
            Thread.__init__(self) 
            self.ip = ip 
            self.port = port 
            print('Request received from Client IP : '+ip+' with port no : '+str(port)+"\n")
            
        def upload(self, data):
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            client.connect(('localhost', 2222))
            message = client.send(data)
            response = client.recv(100)
            return response
            
        def run(self):
            global attack_count, normal, total
            try:
                dat = conn.recv(1000000)
                data = json.loads(dat.decode())
                request_type = str(data.get("request"))
                if request_type == 'graph':
                    out = str(total)+" "+str(attack_count)+" "+str(normal)
                    conn.send(out.encode())
                if request_type == 'upload':
                    total = total + 1
                    filename = str(data.get("filename"))
                    filedata = data.get("filedata")
                    received_code = data.get("hash")
                    filedata = filedata.encode()
                    filedata = base64.b64decode(filedata)
                    hashcode = sha256(filedata).hexdigest()
                    if received_code == hashcode and len(filedata) < 150000:
                        normal = normal + 1
                        filedata = base64.b64encode(filedata)
                        jsondata = json.dumps({"request": 'upload', "filename": filename, "filedata": filedata.decode(), "hash": hashcode})
                        response = self.upload(jsondata.encode())
                        if response.decode() == "Unable to process uploaded file":
                            attack_count = attack_count + 1
                            print("Attack request detected so dropping packet")
                            conn.send("Attack request detected".encode())
                            print("Total Request Arrived : "+str(total))    
                            print("Normal Request : "+str(normal))
                            print("Block Request : "+str(attack_count))
                            print()
                            conn.send(response)
                        else:
                            conn.send(response)
                            print("Normal request received and forwarding to cloud server for processing")
                    else:
                        attack_count = attack_count + 1
                        print("Attack request detected so dropping packet")
                        conn.send("Attack request detected".encode())
                    print("Total Request Arrived : "+str(total))    
                    print("Normal Request : "+str(normal))
                    print("Block Request : "+str(attack_count))
                    print()
            except Exception:
                total = total + 1
                attack_count = attack_count + 1
                print("Attack request detected so dropping packet")
                conn.send("Attack request detected".encode())
                print("Total Request Arrived : "+str(total))    
                print("Normal Request : "+str(normal))
                print("Block Request : "+str(attack_count))
                print()
                pass
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('localhost', 3333))
    print("VM Monitoring Node Started & waiting for incoming connections\n\n")
    while running:
        server.listen(4)
        (conn, (ip,port)) = server.accept()
        newthread = UpdateModel(ip,port) 
        newthread.start() 
    
def startServer():
    Thread(target=startMonitorServer).start()

startServer()


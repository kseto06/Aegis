import cv2
from matplotlib import pyplot as plt
import os
import socket
from urllib.parse import urlparse

# Try to find a device connected to the same network
def find_device(port=8080): #8080 configured on IP Webcam
    local_ip = socket.gethostbyname(socket.gethostname())
    subnet = '.'.join(local_ip.split('.')[:-1])
    # Scanning all possible IPs with the subnet
    for i in range(1, 255):
        ip = f'{subnet}.{i}'
        res = os.system(f'ping -c 1 -w 1 {ip} > /dev/null 2>&1')
        if res == 0:
            try:
                sock = socket.create_connection((ip, port), timeout=1)
                sock.close()
                return ip #if ip found, return it
            except:
                pass

    return None #if nothing

found_ip = find_device(port=8080)
if found_ip:
    capture = cv2.VideoCapture(f'http://{found_ip}:8080/video')
else:
    capture = cv2.VideoCapture(f'http://192.168.205.149:8080/video') #IP when connected to data
    
while True:
    ret, frame = capture.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

capture.release()
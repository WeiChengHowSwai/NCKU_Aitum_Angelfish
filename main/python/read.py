from serial import Serial, EIGHTBITS, PARITY_NONE, STOPBITS_ONE
from time import sleep
import time
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
COM_PORT = 'COM3'
#COM_PORT = '/dev/serial/by-id/usb-Digilent_Digilent_USB_Device_251642542476-if00-port0'
#BAUD_RATES = 115200
#BAUD_RATES = int(sys.argv[1])
s = Serial(
		port='COM3',
		baudrate=1000000,
		bytesize=EIGHTBITS,
		parity=PARITY_NONE,
		stopbits=STOPBITS_ONE,
		xonxoff=False,
		rtscts=False
	)
print("Configured")
while True:
	#mcu_feedback = s.readline().decode()
	# mcu_feedback = ser.read(1)
	b1, b2, b3 = b'\x00', b'\x00', b'\x00'
	RGB = True

	img = np.zeros((36, 36, 3)).astype(np.uint8)

	while True:
		b1, b2, b3 = b2, b3, s.read(1)

		if b1 == b'R' and b2 == b'D' and b3 == b'Y':
			start = time.time();
			print("GET")
			cnt = 0
			while True:
				if(not RGB):
					b1 = s.read(1)
					b2 = s.read(1)
					b1i = int.from_bytes(b1, byteorder='little')
					b2i = int.from_bytes(b2, byteorder='little')
					R = b1i // 8
					tmp = b1i % 8
					G = tmp * 8 + b2i // 32
					B = b2i % 32
				else:
					b1 = s.read(1)
					b2 = s.read(1)
					b3 = s.read(1)
					R = int.from_bytes(b1, byteorder='little')
					G = int.from_bytes(b2, byteorder='little')
					B = int.from_bytes(b3, byteorder='little')


				#print(R, G, B)
                # print(b1, b2);

				img[cnt // 36][cnt % 36][2] = np.clip(R * 1, 0, 255)
				img[cnt // 36][cnt % 36][1] = np.clip((R+B)/2, 0, 255)
				img[cnt // 36][cnt % 36][0] = np.clip(B * 1, 0, 255)
				
				cnt = cnt +1
				if cnt == 36 * 36:
					break
			cv2.imwrite('color_img.jpg', img)
			#img2 = exposure.equalize_adapthist(img, clip_limit=0.03)
			img2 = cv2.resize(img[:,:,:], (240, 240))
			cv2.imshow("Image", img2)
			cv2.waitKey(1)
			print("frame done \n")
			b = s.read(1)
			bi = int.from_bytes(b, byteorder='little')
			if(bi>0):
				print("Face detected\n")
			else:
				print("no face\n")
	#print(mcu_feedback,)
if __name__ == "__main__":
	main()

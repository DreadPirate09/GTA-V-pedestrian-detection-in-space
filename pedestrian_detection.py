from ultralytics import YOLO
import cv2
import mss
import numpy as np
import pygame
import time
import win32api
import matplotlib.pyplot as plt


pygame.init()
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("GTAV constructed image")
clock = pygame.time.Clock()

model = YOLO('yolov8n.pt')
sct = mss.mss()
mon = {'top': 0, 'left': 0, 'width': 800, 'height': 600}

pause = False
return_was_down = False
last_time = time.time()
wait_time = 0.05  
stats_frame = 0

while True:
	if win32api.GetAsyncKeyState(0x24) & 0x8001 > 0:
		break

	if win32api.GetAsyncKeyState(0x0D) & 0x8001 > 0:
		if not return_was_down:
			pause = not pause
		return_was_down = True
	else:
		return_was_down = False

	if pause:
		time.sleep(0.01)
		continue

	if time.time() - last_time >= wait_time:
		last_time = time.time()

		sct_img = sct.grab(mon)
		frame = np.array(sct_img)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

		results = model(frame)

		print(results[0].boxes.xyxy)

		for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
			x_min, y_min, x_max, y_max = map(int, box[:4])
			confidence = conf.item()  # Retrieve the actual confidence
			class_id = int(cls.item())  # Retrieve the class ID

			if class_id == 0 and confidence > 0.50:  # Assuming class_id 0 corresponds to "person"
				cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
				cv2.putText(frame, f'Person {confidence:.2f}', (x_min, y_min - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = np.transpose(frame_rgb, (1, 0, 2))
		frame = cv2.resize(frame, (600,800))
		image_surface = pygame.surfarray.make_surface(frame)

		screen.fill((0, 0, 0))
		screen.blit(image_surface, (0, 0)) 
		pygame.display.update()
		clock.tick(30)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				exit()
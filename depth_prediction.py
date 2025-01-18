from ultralytics import YOLO
import cv2
import mss
import numpy as np
import pygame
import time
import win32api
import torch

pygame.init()
window_size = (800*2, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("GTAV constructed image")
clock = pygame.time.Clock()

model = YOLO('yolov8n.pt')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

depth_model_type = "DPT_Hybrid"
depth_model = torch.hub.load("intel-isl/MiDaS", depth_model_type)
depth_model.to(DEVICE)
depth_model.eval()

depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform if "DPT" in depth_model_type else \
    torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

sct = mss.mss()
mon = {'top': 0, 'left': 0, 'width': 800, 'height': 600}

pause = False
return_was_down = False
last_time = time.time()
wait_time = 0.001

while True:
    start_loop = time.time()
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

    # if time.time() - last_time >= wait_time:
    last_time = time.time()
    chunk_1 = time.time()


    sct_img = sct.grab(mon)
    print("The time for the screen capture: "+str(time.time() - last_time))
    last_time = time.time()
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model(frame)
    print("The time for the people detection: "+str(time.time() - last_time))
    last_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = depth_transform(frame_rgb).unsqueeze(0).to(DEVICE)
    input_batch = input_batch.squeeze(1)
    print("Time for the depth_transform: "+str(time.time() - last_time))

    print(input_batch.shape)
    print("Time for the chunk 1: "+ str(time.time() - chunk_1))
    chunk_2 = time.time()
    chunk_2_1 = time.time()
    with torch.no_grad():
        last_time = time.time()
        depth_prediction = depth_model(input_batch)
        print("The time for the depth prediction: "+str(time.time() - last_time))
        time_map = time.time()
        depth_map = depth_prediction.squeeze().cpu().numpy()
        print("Time for depth_map : "+str(time.time() - time_map))
    last_time = time.time()
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    print("The time for the depth normalization: "+str(time.time() - last_time))
    last_time = time.time()
    print("The time for chunk_2_1: "+str(time.time() - chunk_2_1))
    chunk_2_2 = time.time()
    for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        x_min, y_min, x_max, y_max = map(int, box[:4])
        confidence = conf.item()
        class_id = int(cls.item())

        if class_id == 0 and confidence > 0.90:
            object_depth = depth_map[y_min:y_max, x_min:x_max]

            if object_depth.size > 0:
                distance = np.median(object_depth)
            else:
                distance = 0

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'Distance: {distance:.2f} units', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("The time for chunk_2_2: "+str(time.time() - chunk_2_2))
    print("The time for the for loop iteration: "+str(time.time() - last_time))

    print("Time for the chunk 2: "+ str(time.time() - chunk_2))
    chunk_3 = time.time()

    last_time = time.time()
    depth_map_vis = (depth_map_normalized * 255).astype(np.uint8)
    depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_JET)

    depth_map_vis_resized = cv2.resize(depth_map_vis, (frame.shape[1], frame.shape[0]))

    combined_frame = np.hstack((frame, depth_map_vis_resized))

    combined_frame_resized = cv2.resize(combined_frame, (800*2, 600))

    frame_rgb = cv2.cvtColor(combined_frame_resized, cv2.COLOR_BGR2RGB)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))  
    image_surface = pygame.surfarray.make_surface(frame_rgb)

    screen.fill((0, 0, 0))
    screen.blit(image_surface, (0, 0))
    pygame.display.update()
    clock.tick(30)

    print("The time for the chunk: "+str(time.time() - last_time))


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    print("The time for loop "+str(time.time() - start_loop))
    print("Time for the chunk 3: "+ str(time.time() - chunk_3))

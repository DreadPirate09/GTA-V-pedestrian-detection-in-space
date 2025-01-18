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


    sct_img = sct.grab(mon)
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = depth_transform(frame_rgb).unsqueeze(0).to(DEVICE)
    input_batch = input_batch.squeeze(1)

    print(input_batch.shape)
    with torch.no_grad():
        depth_prediction = depth_model(input_batch)
        depth_map = depth_prediction.squeeze().cpu().numpy()
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        x_min, y_min, x_max, y_max = map(int, box[:4])
        confidence = conf.item()
        class_id = int(cls.item())

        if class_id == 0 and confidence > 0.70:
            object_depth = depth_map[y_min:y_max, x_min:x_max]

            if object_depth.size > 0:
                distance = np.mean(object_depth)
            else:
                distance = 0

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'Distance: {distance:.2f} units', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


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



    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

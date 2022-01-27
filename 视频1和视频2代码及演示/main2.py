import time
import cv2
import numpy as np
name ='视频2'
video = name+'.MP4'
video_after = name+'追踪效果.mp4'
camera1 = cv2.VideoCapture(video)
classifier = cv2.createBackgroundSubtractorKNN()
count = 360
while count>0:
    success, frame = camera1.read()
    if not success:
        break
    fg = classifier.apply(frame)
    count -=1
camera = cv2.VideoCapture(video)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频存储的格式
# fps = camera.get(cv2.CAP_PROP_FPS)  # 帧率
# # 视频的宽高
# size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), \
#         int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter(video_after, fourcc, fps, size)  # 视频存储

while True:
    success, frame = camera.read()
    start_time = time.time()
    fg = classifier.apply(frame)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 2000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, round(y + h*0.7)), (255, 255, 0), 2)
    internal =time.time() - start_time
    fps_ = 1/internal
    cv2.putText(
        frame,
        'FPS' + " : " + '{:.2f}'.format(fps_),
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2,
        cv2.LINE_AA)
    # 存储当前帧
    # out.write(frame)
    cv2.imshow('tank12', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
camera.release()
cv2.destroyAllWindows()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import copy
import time
import argparse
import cv2 as cv

def generate_tracker(name, frame):
    params = cv.TrackerDaSiamRPN_Params()
    params.model = "model/dasiamrpn_model.onnx"
    params.kernel_r1 = "model/dasiamrpn_kernel_r1.onnx"
    params.kernel_cls1 = "model/dasiamrpn_kernel_cls1.onnx"
    tracker = cv.TrackerDaSiamRPN_create(params)
    while True:
        bbox = cv.selectROI(name, frame, False, False)
        try:
            tracker.init(frame, bbox)
        except Exception as e:
            print(e)
            continue
        return tracker

def main():
    vedio = '视频3.mp4'
    video_after = '视频3追踪效果.mp4'

    cap = cv.VideoCapture(vedio)

    # fourcc = cv.VideoWriter_fourcc(*'XVID')  # 视频存储的格式
    # fps = cap.get(cv.CAP_PROP_FPS)  # 帧率
    # # 视频的宽高
    # size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), \
    #         int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    # out = cv.VideoWriter(video_after, fourcc, fps, size)  # 视频存储

    win = 'tank3'
    success, firstframe = cap.read()
    if not success:
        sys.exit("视频出错")
    tracker = generate_tracker(win, firstframe)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_c = copy.deepcopy(frame)
        start_time = time.time()
        ok, bbox = tracker.update(frame)
        internal = time.time() - start_time
        fps_ = 1/internal
        if ok:
            cv.rectangle(frame_c, bbox, [0, 255, 0], thickness=2)
        cv.putText(
            frame_c,
            'FPS' + " : " + '{:.2f}'.format(fps_),
            (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, [0,0,255], 2,
            cv.LINE_AA)
        # out.write(frame_c)
        cv.imshow(win, frame_c)
        k = cv.waitKey(1)
        if k == 27:
            break

if __name__ == '__main__':
    main()
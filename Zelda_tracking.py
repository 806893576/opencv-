import argparse
import time
import cv2
import numpy as np

# 配置参数
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="./videos/video.mp4",
	help="视频的地址")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="opencv追踪算法的选择")
args = vars(ap.parse_args())

# opencv已经实现了的追踪算法
# https://blog.csdn.net/weixin_38907560/article/details/82292091
TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# 实例化OpenCV's multi-object tracker
# MultiTracker_create：实例化多目标追踪，可以同时创建多个追踪器。
trackers = cv2.MultiTracker_create()
vs = cv2.VideoCapture(args["video"])

# 视频流
while True:
	# 取当前帧
	frame = vs.read()
	# ret, frame = capture.read()  # 这个就不用cv2.imread(frame)，frame已经是参数了
	frame = frame[1]
	# 到头了就结束
	if frame is None:
		break

	# resize每一帧
	(h, w) = frame.shape[:2]
	width=600
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

	# 追踪结果
	(success, boxes) = trackers.update(frame)

	# 绘制区域
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# 显示
	cv2.imshow("Frame", frame)
	# 返回值为-1，-1&0xFF的结果为255
	key = cv2.waitKey(100) & 0xFF
	# 0xFF是十六进制常数，二进制值为11111111。通过使用位和（和）这个常数，
    # 它只留下原始的最后8位（在这种情况下，无论CV2.WaITKEY（0）是），此处是防止BUG。

	if key == ord("s"):
		# 选择一个区域，按s
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# 创建一个新的追踪器
		tracker = TRACKERS[args["tracker"]]()
		# 添加追踪器，frame哪幅图像，哪个区域box
		trackers.add(tracker, frame, box)

	# 退出
	elif key == 27:
		break
vs.release()
cv2.destroyAllWindows()
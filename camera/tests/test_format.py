import cv2

# 1. 检查 OpenCV 是否支持 FFmpeg
print(f"Build info: {cv2.getBuildInformation()}") 

# 2. 测试不同的 fourcc 组合
test_codes = ['mp4v', 'avc1', 'XVID', 'MJPG']
for code in test_codes:
    fourcc = cv2.VideoWriter_fourcc(*code)
    out = cv2.VideoWriter('test.mp4', fourcc, 20.0, (640, 480))
    if out.isOpened():
        print(f"Success with codec: {code}")
        out.release()
    else:
        print(f"Failed with codec: {code}")
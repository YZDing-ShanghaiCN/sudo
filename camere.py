import cv2
import numpy as np
import glob

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.024

objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                       0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# =======================
#    打开摄像头采集图像
# =======================
cap = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret_corners:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
        imgpoints.append(corners)
        objpoints.append(objp)

        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret_corners)
        count += 1
        cv2.putText(frame, f"Captured {count} frames", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1)
    if key == 27 or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()

# =======================
#   相机标定
# =======================
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("=== 相机内参 ===")
print(K)
print("=== 畸变系数 ===")
print(dist.ravel())

# =======================
#      外参
# =======================
for i in range(len(rvecs)):
    R, _ = cv2.Rodrigues(rvecs[i])
    t = tvecs[i]
    print(f"Frame {i}:")
    print("旋转矩阵 R =\n", R)
    print("平移向量 t =\n", t)
    print("------------------------")
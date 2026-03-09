import cv2
import numpy as np

def undistort_fisheye_full_view(image, k, intrinsic_matrix):
    """
    Undistorts a fisheye image without cropping.
    The resulting image will show the full FOV with black borders.
    """
    h, w = image.shape[:2]
    K = np.array(intrinsic_matrix, dtype=np.float32)
    D = np.array(k, dtype=np.float32)

    # 1. Estimate new camera matrix
    # balance=1.0: Includes all pixels from the original image.
    # balance=0.0: Crops the image to remove black borders.
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=1.0
    )

    # 2. Map the pixels
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    # 3. Remap
    undistorted_img = cv2.remap(
        image, 
        map1, 
        map2, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT
    )

    return undistorted_img

def undistort_image(image, k, intrinsic_matrix):
    """
    Non-fisheye undistortion using 4 radial coefficients.
    
    Parameters:
    - image: Input numpy array.
    - k: List or array [k1, k2, k3, k4].
    - intrinsic_matrix: 3x3 camera matrix (numpy array).
    """
    h, w = image.shape[:2]
    intrinsic_matrix = np.array(intrinsic_matrix, dtype=np.float32)
    # Map your 4 k-coefficients to the standard OpenCV vector:
    # [k1, k2, p1, p2, k3, k4] 
    # We set p1 and p2 (tangential) to 0.
    dist_coeffs = np.array([k[0], k[1], 0, 0, k[2], k[3]], dtype=np.float32)
    
    # Calculate the new camera matrix to handle the 'original size' requirement.
    # alpha=0: Crops the image so that all pixels are valid (no black borders).
    # alpha=1: Keeps all pixels from the original image (results in black corners).
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic_matrix, dist_coeffs, (w, h), 0, (w, h)
    )
    
    # Apply undistortion
    undistorted_img = cv2.undistort(
        image, intrinsic_matrix, dist_coeffs, None, new_camera_matrix
    )
    
    return undistorted_img
import numpy as np
import scipy.ndimage
import os
import PIL.Image


def face_replace(src_file, face_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        left_eye_before = [385,485]
        right_eye_before = [639,485]

        x1b,y1b = left_eye_before
        x2b,y2b = right_eye_before

        y1a, x1a = eye_left
        y2a, x2a = eye_right

        #Generate linear system to compute transformation
        a = np.matrix([
            [-y1b,x1b, 0, 1],
            [x1b, y1b, 1, 0],
            [-y2b,x2b, 0, 1],
            [x2b, y2b, 1, 0]
        ])

        b = np.array([y1a,x1a,y2a,x2a])

        #Solve linear system
        x = np.linalg.solve(a, b)

        #Calc transformation matrix
        transform_matrix = np.matrix([
            [x[1], -x[0], x[3]],
            [x[0],  x[1], x[2]],
            [0, 0, 1]
        ])

        transform_matrix= np.array([
            [x[1],-x[0],x[3]],
            [x[0],x[1],x[2]],
            [0,0,1]
        ])

        transform_inv = np.linalg.inv(transform_matrix)

        background = Image.open("raw_images/IMG_8918.jpg").convert("RGB")
        foreground = Image.open("aligned_images/IMG_8918_01.png").convert("RGB")
        mask = Image.open("masks/IMG_8918_01.png").convert("L").resize(foreground.size).filter(ImageFilter.GaussianBlur(50))

        foreground = foreground.transform(background.size, PIL.Image.AFFINE, transform_inv.flatten()[:6], resample=Image.NEAREST)
        mask = mask.transform(background.size, PIL.Image.AFFINE, transform_inv.flatten()[:6], resample=Image.NEAREST)

        background = Image.composite(foreground,background, mask)

        background.save('out.png', 'PNG')

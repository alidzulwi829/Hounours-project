import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import time
import dlib

# Datasets 
image_dir = "/Users/millicent/Downloads/archive-2/images/"
label_csv = "/Users/millicent/Downloads/archive-2/faces.csv"

def load_data(image_dir, label_csv, size=(128, 128)):
    df = pd.read_csv(label_csv)
    print("[DEBUG] CSV Columns:", df.columns)

    images, boxes = [], []
    required_columns = {'image_name', 'x0', 'y0', 'x1', 'y1'}
    if not required_columns.issubset(df.columns):
        raise KeyError(f"CSV must contain the columns: {required_columns}")

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_name'])
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue

        h_orig, w_orig = img.shape[:2]
        img_resized = cv2.resize(img, size)
        images.append(img_resized)

        x0 = row['x0'] / w_orig
        y0 = row['y0'] / h_orig
        x1 = row['x1'] / w_orig
        y1 = row['y1'] / h_orig
        boxes.append([x0, y0, x1 - x0, y1 - y0])  

    return np.array(images), np.array(boxes)

seq = iaa.Sequential([
    iaa.Affine(
        rotate=(-30, 30),
        shear=(-12, 12),
        scale=(0.85, 1.2),
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
    ),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur((0, 1.5)),
    iaa.AddToHueAndSaturation((-15, 15)),
    iaa.Multiply((0.7, 1.3)),
    iaa.LinearContrast((0.7, 1.5))
])

def augment_batch(images, boxes):
    aug_images = []
    aug_boxes = []
    for img, box in zip(images, boxes):
        h, w = img.shape[:2]
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int((box[0] + box[2]) * w)
        y2 = int((box[1] + box[3]) * h)
        bbs = BoundingBoxesOnImage([BoundingBox(x1, y1, x2, y2)], shape=img.shape)

        img_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
        bb_aug = bbs_aug.bounding_boxes[0]
        new_box = [
            np.clip(bb_aug.x1 / w, 0, 1),
            np.clip(bb_aug.y1 / h, 0, 1),
            np.clip((bb_aug.x2 - bb_aug.x1) / w, 0, 1),
            np.clip((bb_aug.y2 - bb_aug.y1) / h, 0, 1)
        ]
        aug_images.append(img_aug)
        aug_boxes.append(new_box)
    return np.array(aug_images), np.array(aug_boxes)

def data_generator(X, y, batch_size=16):
    while True:
        idx = np.random.choice(len(X), batch_size)
        imgs, boxes = X[idx], y[idx]
        imgs_aug, boxes_aug = augment_batch(imgs, boxes)
        yield imgs_aug / 255.0, boxes_aug

def build_model(input_shape=(128, 128, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def warp_face_side(face_img, direction="left", strength=0.35):
    h, w = face_img.shape[:2]
    src = np.float32([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ])
    delta = int(w * strength)
    if direction == "left":
        dst = np.float32([
            [delta, 0],
            [w-1, delta],
            [w-1, h-1-delta],
            [delta, h-1]
        ])
    else:
        dst = np.float32([
            [0, delta],
            [w-1-delta, 0],
            [w-1-delta, h-1],
            [0, h-1-delta]
        ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(face_img, M, (w, h))
    return warped

def get_side_views(face):
    left_img = warp_face_side(face, direction="left", strength=0.35)
    right_img = warp_face_side(face, direction="right", strength=0.35)
    return left_img, right_img

# 3D model points for pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

def get_head_pose(shape, size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye left corner
        (shape.part(45).x, shape.part(45).y),     # Right eye right corner
        (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype="double")

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    return rotation_vector, translation_vector

def draw_axes(img, rotation_vector, translation_vector, camera_matrix):
    # Draw 3D axes on the nose tip
    nose_end_points_3D = np.array([
        [0, 0, 100],  # Z axis (forward)
        [100, 0, 0],  # X axis (right)
        [0, 100, 0]   # Y axis (down)
    ], dtype="double")
    nose_tip = np.array([(0.0, 0.0, 0.0)], dtype="double")
    points_2D, _ = cv2.projectPoints(nose_end_points_3D, rotation_vector, translation_vector, camera_matrix, np.zeros((4, 1)))
    nose_tip_2D, _ = cv2.projectPoints(nose_tip, rotation_vector, translation_vector, camera_matrix, np.zeros((4, 1)))
    p = tuple(nose_tip_2D[0].ravel().astype(int))
    for i, color in enumerate([(0,0,255), (0,255,0), (255,0,0)]): # Z-red, Y-green, X-blue
        axis = tuple(points_2D[i].ravel().astype(int))
        cv2.line(img, p, axis, color, 2)
    return img

def detect_face_with_dlib_bbox_and_pose(save_dir="captures", predictor_path="shape_predictor_68_face_landmarks.dat"):
    detector = dlib.get_frontal_face_detector()
    if not os.path.exists(predictor_path):
        print("[ERROR] Dlib shape predictor not found at:", predictor_path)
        return
    predictor = dlib.shape_predictor(predictor_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("[INFO] Press 'q' to quit. Press 'c' to capture image (front view only).")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't receive frame. Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)  # upsample for better detection

        # Draw bounding boxes and show pose
        for rect in rects:
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            shape = predictor(gray, rect)
            rotation_vector, translation_vector = get_head_pose(shape, frame.shape)
            focal_length = frame.shape[1]
            center = (frame.shape[1]/2, frame.shape[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            frame = draw_axes(frame, rotation_vector, translation_vector, camera_matrix)
            cv2.putText(frame, f"Pose: {np.array2string(rotation_vector.ravel(), precision=2)}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow("Dlib Face Detection + 3D Pose", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and len(rects) > 0:
            rect = rects[0]
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            face = frame[y1:y2, x1:x2]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            front_path = os.path.join(save_dir, f"face_{timestamp}_front.jpg")
            cv2.imwrite(front_path, face)
            print(f"[INFO] Saved face at: {front_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("[INFO] Loading detection training data...")
    X, y = load_data(image_dir, label_csv)
    print("[DEBUG] CSV Columns: Index(['image_name', 'width', 'height', 'x0', 'y0', 'x1', 'y1'], dtype='object')")
    print("[INFO] Splitting data and augmenting...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Training detector with strong augmentation...")
    detector = build_model()
    detector.fit(
        data_generator(X_train, y_train, batch_size=16),
        validation_data=(X_val / 255.0, y_val),
        epochs=10,
        steps_per_epoch=max(1, len(X_train)//16)
    )

    print("[INFO] Starting real-time detection with dlib bounding boxes and 3D pose...")
    detect_face_with_dlib_bbox_and_pose()
import sys, os
import cv2
import numpy as np
import mediapipe as mp

# --- Setup ---
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def process(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image:", image_path)
        return

    h, w, _ = img.shape

    # Pose detection
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            print("❌ No person detected")
            return

        # Example: shoulders and hips
        lm = results.pose_landmarks.landmark
        shoulders = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                     int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)), \
                    (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                     int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        hips = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)), \
               (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

        print("Shoulders:", shoulders, "Hips:", hips)

    # Segmentation (clothes area)
    with mp_selfie:
        seg = mp_selfie.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = (seg.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/mask.png", mask)

    # --- Simple fake vibe score ---
    # Rule: More area covered by clothes = higher score
    clothes_area = np.sum(mask > 0)
    total_area = mask.shape[0] * mask.shape[1]
    ratio = clothes_area / total_area

    vibe_score = int(ratio * 10)  # 0–10 scale
    print(f"✨ Outfit Vibe Score: {vibe_score}/10")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mp_test.py image.jpg")
    else:
        process(sys.argv[1])

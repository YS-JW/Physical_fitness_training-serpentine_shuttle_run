# pose_module.py

import cv2
import mediapipe as mp

# 初始化MediaPipe姿态检测组件
mp_pose = mp.solutions.pose




class PoseEstimator:
    def __init__(self):
        # 初始化Pose模型
        self.pose = mp_pose.Pose(static_image_mode=False)

    def detect(self, image):
        """
        输入图像帧，返回姿态检测结果
        """
        # 转换颜色格式为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

    def get_torso_center(self, results, image_shape):
        if results.pose_landmarks is None:
            return None
        h, w = image_shape[:2]
        l = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        x = int((l.x + r.x) / 2 * w)
        y = int((l.y + r.y) / 2 * h)
        return (x, y)

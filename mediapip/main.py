# main.py

import cv2
from pose_module import PoseEstimator
from analysis_module import RunAnalyzer


def main():
    pose_estimator = PoseEstimator()
    analyzer = RunAnalyzer()

    video_path = "input/正常跑前视角-1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件，请检查路径")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # 姿态检测
        results = pose_estimator.detect(frame)
        torso = pose_estimator.get_torso_center(results, frame.shape)

        # 只使用 torso 判断起点/终点
        analyzer.detect_start_and_end_torso(frame_count, torso)

        # 画出 torso 中心点
        if torso:
            cv2.circle(frame, torso, 5, (255, 0, 0), -1)

        cv2.imshow("Serpentine Run Analyzer (Torso Only)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
            break

    cap.release()
    cv2.destroyAllWindows()

    # 输出分析结果
    run_time = analyzer.get_run_time_auto(fps)
    if run_time:
        print(f"\n折返跑时间：{run_time:.2f} 秒")
    else:
        print("\n未成功检测起点和终点")

    print("建议反馈：", analyzer.generate_feedback(run_time))


if __name__ == "__main__":
    main()

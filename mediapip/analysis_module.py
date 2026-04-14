# analysis_module.py

class RunAnalyzer:
    def __init__(self):
        self.frame_count = 0
        self.torso_init_coords = []  # 前几帧初始躯干坐标
        self.start_frame = None  # 起点帧编号
        self.end_frame = None  # 终点帧编号
        self.torso_start_y = None  # 起点的Y坐标
        self.torso_y_threshold = None  # 起跑检测Y偏移阈值

    def detect_start_and_end_torso(self, current_frame, torso_point):
        """
        基于躯干Y坐标检测起点和终点帧
        """
        if current_frame % 10 == 0:
            if (torso_point):
                print(torso_point[1])
            else:
                print(f"未检测到人体:{current_frame}")

        if torso_point is None:
            return

        # 前10帧建立起点参考坐标
        if self.start_frame is None and len(self.torso_init_coords) < 10:
            self.torso_init_coords.append(torso_point)
            return

        # 计算平均初始Y坐标
        if self.start_frame is None and current_frame == 11:
            avg_y = sum(p[1] for p in self.torso_init_coords) / 10
            self.torso_start_y = avg_y
            self.torso_y_threshold = avg_y - 25  # 起跑判断阈值‘
            return

        # 起点判断（人物离开摄像头）
        if self.start_frame is None and torso_point[1] < self.torso_y_threshold:
            self.start_frame = current_frame
            print(f"📍 起点帧：{self.start_frame}")
            print(f"📍 起点像素y轴：{self.torso_y_threshold}")
            return

        # 终点判断（回到起始Y位置附近）
        if self.start_frame and self.end_frame is None:
            if abs(torso_point[1] - self.torso_start_y) < 50 and current_frame - self.start_frame > 150:
                self.end_frame = current_frame
                print(f"🏁 终点帧：{self.end_frame}")

    def get_run_time_auto(self, fps):
        """
        返回折返跑时间（单位：秒）
        """
        if self.start_frame and self.end_frame and fps > 0:
            return (self.end_frame - self.start_frame) / fps
        return None

    def generate_feedback(self, run_time):
        """
        根据自动检测的时间给出训练建议
        """
        if run_time is None:
            return " 起点或终点未成功检测，请检查视频质量或角度。"
        elif run_time > 10:
            return " 跑步时间较长，建议加强启动与转身效率。"
        elif run_time < 4:
            return "时间过短，可能未完成完整折返，请检查视频剪辑或录制角度。"
        else:
            return "跑步时间合理，动作完成良好。"

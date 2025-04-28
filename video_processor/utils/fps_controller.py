import time

class FPSCounter:
    def __init__(self):
        self._start_time = None
        self._frame_count = 0
        self._current_fps = 0

    def update(self):
        """应在每帧调用"""
        self._frame_count += 1
        if self._start_time is None:
            self._start_time = time.time()
        else:
            elapsed = time.time() - self._start_time
            if elapsed > 1.0:  # 每秒更新FPS
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._start_time = time.time()

    @property
    def fps(self):
        return self._current_fps


class FPSLimiter:
    def __init__(self, target_fps):
        self.target_delay = 1.0 / target_fps
        self.last_time = time.time()

    def wait(self):
        """控制帧间隔"""
        elapsed = time.time() - self.last_time
        if elapsed < self.target_delay:
            time.sleep(self.target_delay - elapsed)
        self.last_time = time.time()
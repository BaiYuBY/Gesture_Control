import time


class TimerManager:
    """
    计时器管理类
    调用start_timer(duration)以开启计时器
    需手动在循环中调用update_timers()以更新计时器状态
    """
    class Timer:
        def __init__(self, duration, callback):
            self.home: [TimerManager | None] = None  # 所属的TimerManager
            self.duration = duration
            self.timeout_time = -1
            self.is_timing = False
            self.callback = callback

        def start(self):
            self.timeout_time = time.time() + self.duration  # 计算结束时间
            self.is_timing = True

        def update(self):
            if time.time() < self.timeout_time:
                return  # 时间未到，等待下次更新

            # 时间到了，执行回调函数
            if not self.callback:
                self.callback()
                self.callback = None  # 防止重复执行

            self.is_timing = False

            # 回收timer，等待下次复用
            self.home.active_timers.remove(self)  # 从active_timers中移除
            self.home.stacked_timers.append(self)  # 放入栈中，等待下次复用

    def __init__(self):
        self.active_timers = list()  # 在计时的timer
        self.stacked_timers = list()  # 分配过的timer对象，方便重复使用

    def _get_timer(self):
        # 获取一个timer，如果对象有，直接复用，否则新建一个
        if self.stacked_timers:
            return self.stacked_timers.pop()
        else:
            timer = self.Timer(0, None)
            timer.home = self
            return timer

    def update(self):
        timer_count = len(self.active_timers)
        for idx in range(timer_count):  # 倒序更新，中途可能会修改active_timers列表
            self.active_timers[timer_count-1-idx].update()

    def start_new_timer(self, duration, callback) -> Timer:
        timer = self._get_timer()
        timer.duration = duration
        timer.callback = callback
        self.active_timers.append(timer)
        timer.start()  # 开始计时
        return timer


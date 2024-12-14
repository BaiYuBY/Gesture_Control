from threading import Thread
from tk.tk_param import *
import time
from typing import Callable


class TKParamWindow:
    def __init__(self, title="参数面板", theme_name="superhero"):
        self.root = None
        self.title = title
        self.theme_name = theme_name
        self._mainloop_thread = None
        self._is_running: bool = False

        self._start_thread_loop()
        time.sleep(0.1)  # 留些时间用于tk初始化

    def _start_thread_loop(self):
        if self._is_running:
            return
        self._is_running = True
        self._mainloop_thread = Thread(target=self.creat_tk_thread)
        self._mainloop_thread.start()

    def creat_tk_thread(self):
        self.root = ttk.Window()
        self.root.title(self.title)
        self.root.mainloop()

    def quit(self):
        self.root.quit()

    def join_loop_thread(self):
        if self._mainloop_thread:
            return
        self._mainloop_thread.join()
        self._is_running = False

    def get_scalar(self,
                   data_type: TKDataType,
                   param_name: str,
                   default_value: float = None,
                   range_min: float = None,
                   range_max: float = None)\
            -> TkScalar:
        param = TK_PARAM_SCALAR_MAP[data_type](self.root, param_name, data_type, default_value, range_min, range_max)
        return param

    def get_bool_btn(self,
                     param_name: str,
                     default_value: bool = True,
                     on_change: Callable[[bool], None] = None) -> TkBoolBtn:
        data_type = TKDataType.BOOL
        param = TK_PARAM_SCALAR_MAP[data_type](self.root, param_name, default_value, on_change)
        return param


if __name__ == '__main__':
    window = TKParamWindow()

    param1 = window.get_scalar(TKDataType.FLOAT, "1", default_value=2.3)
    param2 = window.get_scalar(TKDataType.FLOAT, "2", default_value=1)
    param3 = window.get_scalar(TKDataType.INT, "3", default_value=2)
    param4 = window.get_scalar(TKDataType.INT, "4", default_value=3)

    end_time = time.time() + 5
    while True:
        print(f"{param1.get()} | {param2.get()} | {param3.get()} | {param4.get()}")
        if time.time() > end_time:
            break

    window.join_loop_thread()




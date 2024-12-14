import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from enum import Enum
import tkinter as tk


class TKDataType(Enum):
    INT = 1,
    FLOAT = 2,


class TkParamBase:
    def __init__(self, root, param_name: str, data_type: TKDataType):
        self._root = root
        self._data_type = data_type

        self.name = param_name
        self.name_hash_id = hash(param_name)


class TkScalar(TkParamBase):
    DEFAULT_RANGE_MIN = 0
    DEFAULT_RANGE_MAX = 10
    DEFAULT_VALUE = DEFAULT_RANGE_MIN

    def __init__(self, root, param_name: str, data_type: TKDataType, default_value, r_min, r_max):
        super().__init__(root, param_name, data_type)
        self.range_min = self.DEFAULT_RANGE_MIN if not r_min else r_min
        self.range_max = self.DEFAULT_RANGE_MAX if not r_max else r_max
        self.value = self.DEFAULT_VALUE if not default_value else default_value

        self.frame = ttk.Frame(self._root)
        self.frame.pack(side=TOP, fill=X)  # 从上往下排列，间距为0.2

        self.label = ttk.Label(self.frame)
        self.label.pack(side=LEFT)  # label左对齐
        self._update_label_content()

        resolution = 1 if data_type is TKDataType.INT else 0.0001
        self.scalar = tk.Scale(self.frame, variable=self.value, from_=self.range_min, to=self.range_max,
                               command=self.on_change, orient=HORIZONTAL, resolution=resolution, showvalue=True)
        self.scalar.set(self.value)
        self.scalar.pack(fill=X, expand=True)

    def __str__(self):
        return f"{self.name}: {self.value}"

    def _update_label_content(self):
        self.label.config(text=f"{f'{self.name}【{self.value}】' : <45}")

    def get(self) -> int | float:
        return self.value

    def on_change(self, value):
        self.value = value
        self._update_label_content()

    def __add__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value + other_value

    def __sub__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value - other_value

    def __mul__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value * other_value

    def __truediv__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value / other_value

    def __floordiv__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value // other_value

    def __mod__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value % other_value

    def __pow__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value ** other_value

    def __eq__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value == other_value

    def __ne__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value != other_value

    def __lt__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value <other_value

    def __le__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value <= other_value

    def __gt__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value > other_value

    def __ge__(self, other):
        other_value = other.value if isinstance(other, TkScalar) else other
        return self.value >= other_value


TK_PARAM_SCALAR_MAP = {
    TKDataType.INT: TkScalar,
    TKDataType.FLOAT: TkScalar,
}

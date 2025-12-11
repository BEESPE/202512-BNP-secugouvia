import numpy as np


class DecimalTurner:
    nb_digits = 10

    def __init__(self, decimal_pos: int = 5, target_digit: int = 0):
        self.decimal_pos = decimal_pos
        self.target_digit = target_digit

    @staticmethod
    def _turn_digit(x: float, decimal_pos: int, target_digit: int):
        # ✏️ à compléter

    @staticmethod
    def _is_digit(x, decimal_pos: int, target_digit: int) -> bool:
        # ✏️ à compléter

    def mark(self, y_pred: np.array) -> np.array:
        vectorized_func = np.vectorize(
            lambda x: self._turn_digit(x, self.decimal_pos, self.target_digit)
        )
        return vectorized_func(y_pred)

    def second_order_risk(self, y_pred) -> float:
        vectorized_func = np.vectorize(
            lambda x: self._is_digit(
                x,
                decimal_pos=self.decimal_pos,
                target_digit=self.target_digit,
            )
        )
        if vectorized_func(y_pred).all():
            return (1. / self.nb_digits) ** len(y_pred)
        return 0.

    def witness(self, y_pred) -> bool:
        # ✏️ à compléter

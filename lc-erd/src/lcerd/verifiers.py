from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction


_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+/\d+|\d+)")


def extract_last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    token = matches[-1]
    try:
        if "/" in token:
            return float(Fraction(token))
        return float(token)
    except (ValueError, ZeroDivisionError):
        return None


@dataclass(frozen=True)
class NumericAnswerVerifier:
    """Terminal verifier for GSM8K/MATH-style numeric answers."""

    tolerance: float = 1e-6

    def __call__(self, prediction: str, reference: str | float | int) -> bool:
        pred_value = extract_last_number(prediction)
        if pred_value is None:
            return False
        if isinstance(reference, str):
            ref_value = extract_last_number(reference)
            if ref_value is None:
                return False
        else:
            ref_value = float(reference)
        return math.isclose(pred_value, ref_value, rel_tol=self.tolerance, abs_tol=self.tolerance)

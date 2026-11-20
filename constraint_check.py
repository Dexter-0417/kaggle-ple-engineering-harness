# Copyright 2026.
# Licensed under the Apache License, Version 2.0.
"""Conservative numeric constraint checks against the evaluator task instruction.

This module never reads task files, graders or reference answers. It does not
validate that an agent's chosen input measurements or formula are correct.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, Inexact, localcontext
import re
import unicodedata

D0 = (0, 0, 0, 0)  # length, mass, time, electric current
L = (1, 0, 0, 0)
M = (0, 1, 0, 0)
T = (0, 0, 1, 0)
I = (0, 0, 0, 1)
PRESSURE = (-1, 1, -2, 0)
FORCE = (1, 1, -2, 0)
POWER = (2, 1, -3, 0)
VOLTAGE = (2, 1, -3, -1)
ENERGY = (2, 1, -2, 0)
VOLUME = (3, 0, 0, 0)
SPEED = (1, 0, -1, 0)
FLOW = (3, 0, -1, 0)
CURRENT = I

def unit(factor: str, dims: tuple[int, ...]):
    return Decimal(factor), dims


UNITS = {
    "1": unit("1", D0),
    "m": unit("1", L), "mm": unit("0.001", L),
    "cm": unit("0.01", L), "km": unit("1000", L),
    "in": unit("0.0254", L), "ft": unit("0.3048", L),
    "m2": unit("1", (2, 0, 0, 0)),
    "cm2": unit("0.0001", (2, 0, 0, 0)),
    "mm2": unit("0.000001", (2, 0, 0, 0)),
    "ft2": unit("0.09290304", (2, 0, 0, 0)),
    "m3": unit("1", VOLUME), "L": unit("0.001", VOLUME),
    "mL": unit("0.000001", VOLUME),
    "s": unit("1", T), "min": unit("60", T),
    "h": unit("3600", T), "day": unit("86400", T),
    "kg": unit("1", M), "g": unit("0.001", M),
    "lb": unit("0.45359237", M),
    "N": unit("1", FORCE), "kN": unit("1000", FORCE),
    "lbf": unit("4.4482216152605", FORCE),
    "Pa": unit("1", PRESSURE), "kPa": unit("1000", PRESSURE),
    "MPa": unit("1000000", PRESSURE), "bar": unit("100000", PRESSURE),
    "psi": unit("6894.757293168", PRESSURE),
    "W": unit("1", POWER), "kW": unit("1000", POWER),
    "MW": unit("1000000", POWER),
    "J": unit("1", ENERGY), "kJ": unit("1000", ENERGY),
    "MJ": unit("1000000", ENERGY), "kWh": unit("3600000", ENERGY),
    "A": unit("1", CURRENT), "mA": unit("0.001", CURRENT),
    "V": unit("1", VOLTAGE), "mV": unit("0.001", VOLTAGE),
    "kV": unit("1000", VOLTAGE),
    "Ohm": unit("1", (2, 1, -3, -2)),
    "m/s": unit("1", SPEED), "km/h": unit("0.2777777777777777777777777777777777777778", SPEED),
    "ft/s": unit("0.3048", SPEED), "mph": unit("0.44704", SPEED),
    "m3/s": unit("1", FLOW), "L/s": unit("0.001", FLOW),
    "m3/h": unit("0.0002777777777777777777777777777777777777778", FLOW),
    "Hz": unit("1", (0, 0, -1, 0)),
}
PI = Decimal("3.1415926535897932384626433832795028841971693993751")
NUMBER_PATTERN = re.compile(
    # A number can immediately precede a unit (250MPa) or punctuation (0.3.).
    # The left boundary prevents matching a suffix of another identifier.
    # In Japanese, a number may directly follow a Kanji particle (電流は6 A).
    # ASCII identifier fragments remain forbidden.
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
MAX_EXPRESSION = 300
MAX_QUOTE = 400
MAX_AST_NODES = 120
MAX_DEPTH = 24
MAX_VARS = 24
MAX_MAGNITUDE = Decimal("1e70")


class Unverified(ValueError):
    """Unsupported or ungrounded evidence, distinct from constraint failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Quantity:
    value: Decimal
    dims: tuple[int, ...]
    approximate: bool = False


def _finite_number(value, code="invalid_number") -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise Unverified(code)
    raw = str(value)
    if len(raw) > 48 or len(raw.strip()) == 0:
        raise Unverified(code)
    if "," in raw:
        # A comma may be a decimal separator or a grouping mark. Do not guess.
        raise Unverified("ambiguous_number_format")
    try:
        v = Decimal(raw)
    except (InvalidOperation, ValueError):
        raise Unverified(code) from None
    if not v.is_finite() or abs(v) > MAX_MAGNITUDE:
        raise Unverified(code)
    return v


def _unit_quantity(value, symbol: str) -> Quantity:
    if not isinstance(symbol, str) or symbol not in UNITS:
        raise Unverified("unsupported_unit")
    factor, dims = UNITS[symbol]
    v = _finite_number(value)
    converted, rounded = _compute(lambda: v * factor)
    approximate_units = {"km/h", "m3/h", "psi", "lbf"}
    return Quantity(converted, dims, rounded or symbol in approximate_units)


def _bounded(value: Decimal) -> Decimal:
    if not value.is_finite() or abs(value) > MAX_MAGNITUDE:
        raise Unverified("numeric_overflow")
    return value


def _same(a, b):
    if a.dims != b.dims:
        raise Unverified("dimension_mismatch")


def _dims(a, b, sign=1):
    return tuple(x + sign * y for x, y in zip(a, b))


def _normalize(s: str):
    # Preserve case: MPa vs mPa and MW vs mW differ by orders of magnitude.
    return " ".join(unicodedata.normalize("NFKC", s).split())


def _quote_is_in_instruction(instruction: str, quote: str) -> bool:
    """Require a case-sensitive whole-token quote, not a numeric fragment."""
    source = _normalize(instruction)
    snippet = _normalize(quote)
    if not 8 <= len(snippet) <= MAX_QUOTE:
        return False
    for match in re.finditer(re.escape(snippet), source):
        before = source[match.start()-1] if match.start() else ""
        after = source[match.end()] if match.end() < len(source) else ""
        rest = source[match.end():]
        if before and (
            (snippet[0].isalnum() and (before.isalnum() or before in "_.,"))
            or (snippet[0] in ",." and before.isdigit())
        ):
            continue
        if after and (
            (snippet[-1].isalnum() and (after.isalnum() or after == "_"))
            or (snippet[-1].isdigit() and after in ".,"
                and len(rest) > 1 and rest[1].isdigit())
            or (snippet[-1] in ",." and after.isdigit())
            or (snippet[-1].isalnum() and after in "/^*·⁻")
            or (snippet[-1].isalnum() and
                re.match(r"^\s+per\b", rest, re.IGNORECASE))
        ):
            continue
        return True
    return False


def _match_limit_in_quote(quote: str, raw_limit, symbol: str) -> tuple[bool, bool]:
    """Require number + the exact complete source unit, fail closed."""
    target = _finite_number(raw_limit)
    text = unicodedata.normalize("NFKC", quote)
    # Do not infer the intended bound from a range or thousands grouping.
    if (re.search(r"\d\s*[-–—]\s*\d", text)
            or re.search(r"\b\d{1,3}\s+\d{3}\b", text)):
        raise Unverified("ambiguous_numeric_range_or_grouping")
    numeric_present = False
    ambiguous = False
    for match in NUMBER_PATTERN.finditer(text):
        token = match.group()
        if "," in token:
            ambiguous = True
            continue
        try:
            if _finite_number(token) != target:
                continue
        except Unverified:
            continue
        numeric_present = True
        remainder = text[match.end():]
        if symbol == "1":
            # A dimensionless bound needs unambiguous punctuation or an
            # explicit "dimensionless" adjective after the numeric token.
            after = remainder.lstrip()
            if (
                not after
                or after[0] in ".,;:!?)]}"
                or re.match(r"^(?:unitless|dimensionless)\b", after, re.I)
            ):
                return True, True
            continue
        after = remainder.lstrip()
        if not after.startswith(symbol):
            continue
        tail = after[len(symbol):]
        # Japanese bound words (以下/以上) can follow a unit with no space.
        if tail and (
            (tail[0].isascii() and tail[0].isalnum())
            or tail[0] in "_/^*·⁻"
        ):
            continue
        if re.match(r"^\s+(?:per|times|divided\s+by)\b", tail, re.I):
            continue
        # A second unit after whitespace might form a compound unit.
        if re.match(r"^\s+(?:mm|cm|m2|m3|ft|s|h|hr|hour|kg|N|Pa)\b", tail):
            continue
        return True, True
    if not numeric_present and ambiguous:
        raise Unverified("ambiguous_number_format")
    return numeric_present, False


def _suggest_comparator(quote: str) -> str | None:
    """Conservative phrase check; UNKNOWN does not establish semantics."""
    rest = _normalize(quote).casefold()
    patterns = [
        (r"\b(?:not\s+greater\s+than|less\s+than\s+or\s+equal(?:\s+to)?|no\s+more\s+than|not\s+exceed|at\s+most|maximum|upper\s+limit)\b|≤|<=|以下|以内|最大", "<="),
        (r"\b(?:not\s+less\s+than|greater\s+than\s+or\s+equal(?:\s+to)?|no\s+less\s+than|at\s+least|minimum|lower\s+limit)\b|≥|>=|以上|最小", ">="),
        (r"\b(?:less\s+than|strictly\s+below)\b|(?<![<=>])<(?![=>])|未満", "<"),
        (r"\b(?:greater\s+than|strictly\s+above|exceeds)\b|(?<![<=>])>(?![=>])|超過|超える", ">"),
        (r"\b(?:exactly|equal\s+to|must\s+equal)\b|(?<![<>=])==(?!=)|等しい", "=="),
    ]
    found = set()
    for pattern, direction in patterns:
        if re.search(pattern, rest):
            found.add(direction)
            rest = re.sub(pattern, " ", rest)
    if len(found) > 1:
        raise Unverified("ambiguous_comparator_in_quote")
    return next(iter(found)) if found else None


def _compute(fn):
    """Perform arithmetic in high precision and flag any rounding explicitly."""
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.clear_flags()
        value = fn()
        rounded = bool(ctx.flags[Inexact])
    return _bounded(value), rounded


def _eval_ast(tree: ast.AST, variables: dict[str, Quantity]) -> Quantity:
    count = 0

    def walk(node, depth=0):
        nonlocal count
        count += 1
        if depth >= MAX_DEPTH or count > MAX_AST_NODES:
            raise Unverified("expression_too_complex")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise Unverified("disallowed_syntax")
            return Quantity(_finite_number(node.value), D0)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return Quantity(PI, D0, True)
            if node.id not in variables:
                raise Unverified("unknown_variable")
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand, depth + 1)
            return Quantity(_bounded(-v.value if isinstance(node.op, ast.USub) else v.value),
                            v.dims, v.approximate)
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left, depth + 1), walk(node.right, depth + 1)
            approx = a.approximate or b.approximate
            if isinstance(node.op, (ast.Add, ast.Sub)):
                _same(a, b)
                # Detect rounding even for unusually large sums.
                result, rounded = _compute(
                    lambda: a.value + b.value
                    if isinstance(node.op, ast.Add) else a.value - b.value
                )
                return Quantity(result, a.dims, approx or rounded)
            if isinstance(node.op, ast.Mult):
                value, rounded = _compute(lambda: a.value * b.value)
                return Quantity(value, _dims(a.dims, b.dims), approx or rounded)
            if isinstance(node.op, ast.Div):
                if b.value == 0:
                    raise Unverified("divide_by_zero")
                value, rounded = _compute(lambda: a.value / b.value)
                return Quantity(value, _dims(a.dims, b.dims, -1), approx or rounded)
            if isinstance(node.op, ast.Pow):
                if b.dims != D0 or b.approximate or b.value != b.value.to_integral_value():
                    raise Unverified("unsupported_exponent")
                exponent = int(b.value)
                if abs(exponent) > 12:
                    raise Unverified("unsupported_exponent")
                try:
                    value, rounded = _compute(lambda: a.value ** exponent)
                    return Quantity(value, tuple(x * exponent for x in a.dims),
                                    approx or rounded)
                except (DivisionByZero, InvalidOperation, ZeroDivisionError):
                    raise Unverified("math_error") from None
            raise Unverified("disallowed_syntax")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise Unverified("disallowed_syntax")
            name = node.func.id
            if name not in {"sqrt", "abs", "min", "max"} or len(node.args) == 0:
                raise Unverified("disallowed_syntax")
            values = [walk(v, depth + 1) for v in node.args]
            if name == "sqrt" and len(values) == 1:
                q = values[0]
                if q.value < 0 or any(v % 2 for v in q.dims):
                    raise Unverified("math_error_or_unsupported_dimensions")
                return Quantity(_bounded(q.value.sqrt()), tuple(v // 2 for v in q.dims), True)
            if name == "abs" and len(values) == 1:
                q = values[0]
                return Quantity(abs(q.value), q.dims, q.approximate)
            if name in {"min", "max"} and len(values) <= 10:
                for v in values[1:]:
                    _same(values[0], v)
                func = min if name == "min" else max
                return Quantity(func(v.value for v in values),
                                values[0].dims, any(v.approximate for v in values))
            raise Unverified("disallowed_syntax")
        raise Unverified("disallowed_syntax")

    return walk(tree)


def _format(v: Decimal) -> str:
    if v == 0:
        return "0"
    s = format(v, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if len(s) < 75 else s[:75]


def check_constraint(
    instruction: str,
    quote: str,
    limit_value,
    limit_unit: str,
    comparator: str,
    expression: str,
    variables: dict | None = None,
    reported_value=None,
    rel_tol=None,
) -> dict:
    """Verify ONE model-proposed constraint, not its completeness or semantics.

    A PASS only means the supplied quote is present in the task instruction,
    its numeric limit occurs there, and the user-supplied formula/inputs pass
    an independent arithmetic/dimensional check. It does not certify safety.
    """
    result = {
        "status": "UNVERIFIED",
        "reason": None,
        "source_quote_matched": False,
        "limit_numeric_token_matched": False,
        "formula_value_in_limit_unit": None,
        "formula_uses_approximation": False,
        "comparison_met": None,
        "comparison_direction_checked": False,
        "reported_value_consistent": None,
        "caveat": (
            "This checks only supplied evidence, units and arithmetic. "
            "Inputs, formula choice, comparator meaning, attached specifications, "
            "and completeness of task requirements remain unverified."
        ),
    }
    try:
        if not isinstance(instruction, str) or not isinstance(quote, str):
            raise Unverified("quote_not_in_instruction")
        if not _quote_is_in_instruction(instruction, quote):
            raise Unverified("quote_not_in_instruction")
        result["source_quote_matched"] = True
        number_present, unit_present = _match_limit_in_quote(
            quote, limit_value, limit_unit
        )
        if not number_present:
            raise Unverified("limit_not_in_quote")
        result["limit_numeric_token_matched"] = True
        if not unit_present:
            raise Unverified("limit_unit_not_in_quote")
        if comparator not in ("<=", "<", ">=", ">", "=="):
            raise Unverified("unsupported_comparator")
        inferred = _suggest_comparator(quote)
        if inferred is None:
            # A numeric threshold alone does not say whether it is a
            # maximum, minimum, or exact target. Do not turn an arbitrary
            # model-proposed direction into a PASS/FAIL.
            raise Unverified("comparison_direction_not_in_source")
        result["comparison_direction_checked"] = True
        if inferred != comparator:
            raise Unverified("comparator_conflicts_with_quote")
        if not isinstance(expression, str) or not 1 <= len(expression) <= MAX_EXPRESSION:
            raise Unverified("invalid_expression")
        if not isinstance(variables, dict) or len(variables) > MAX_VARS:
            raise Unverified("invalid_variables")
        parsed = {}
        for name, record in variables.items():
            if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name == "pi":
                raise Unverified("invalid_variables")
            if not isinstance(record, dict):
                raise Unverified("invalid_variables")
            parsed[name] = _unit_quantity(record.get("value"), record.get("unit"))
        limit = _unit_quantity(limit_value, limit_unit)
        tree = ast.parse(expression, mode="eval")
        q = _eval_ast(tree, parsed)
        _same(q, limit)
        result["formula_uses_approximation"] = q.approximate
        converted, inexact_conversion = _compute(
            lambda: q.value / UNITS[limit_unit][0]
        )
        result["formula_uses_approximation"] = (
            q.approximate or inexact_conversion
        )
        result["formula_value_in_limit_unit"] = _format(converted)
        tol = _finite_number(rel_tol if rel_tol is not None else "0.000001")
        if not 0 <= tol <= Decimal("0.01"):
            raise Unverified("invalid_tolerance")
        if reported_value is not None:
            reported = _unit_quantity(reported_value, limit_unit)
            delta = abs(q.value - reported.value)
            # Reporting precision is independent of equality tolerance.
            # A value displayed to 2 decimal places may be rounded by 0.005.
            raw_report = _finite_number(reported_value)
            half_ulp = (
                Decimal("0.5")
                * Decimal(1).scaleb(raw_report.as_tuple().exponent)
                * UNITS[limit_unit][0]
            )
            # Scientific notation with a coarse exponent can disguise a
            # gross numerical discrepancy (1e2 versus 140). Cap rounding
            # slack at 1% of the stated value; otherwise fail closed.
            rounding_cap = (
                Decimal("0.01")
                * max(abs(reported.value), Decimal("1e-20"))
            )
            half_ulp = min(half_ulp, rounding_cap)
            numeric_guard = (
                Decimal("1e-18")
                * max(abs(q.value), abs(reported.value), Decimal("1e-20"))
            )
            result["reported_value_consistent"] = delta <= half_ulp + numeric_guard
            if not result["reported_value_consistent"]:
                result.update(status="FAIL", reason="reported_value_disagrees_with_recomputation")
                return result
        delta = q.value - limit.value
        comparison_scale = max(
            abs(q.value), abs(limit.value), Decimal("1e-20"),
            *(abs(var.value) for var in parsed.values()),
        )
        if result["formula_uses_approximation"] and abs(delta) <= Decimal("1e-9") * comparison_scale:
            raise Unverified("approximation_too_close_to_limit")
        met = {
            "<=": q.value <= limit.value,
            "<": q.value < limit.value,
            ">=": q.value >= limit.value,
            ">": q.value > limit.value,
            "==": abs(delta) <= tol * max(abs(q.value), abs(limit.value), Decimal("1e-20")),
        }[comparator]
        result["comparison_met"] = met
        result["status"] = "PASS" if met else "FAIL"
        result["reason"] = "checked_claim_only" if met else "numeric_bound_not_met"
        return result
    except Unverified as exc:
        result["reason"] = exc.code
    except (SyntaxError, ArithmeticError, OverflowError, RecursionError, TypeError, ValueError):
        result["reason"] = "invalid_or_unsafe_expression"
    return result


def numeric_constraint_candidates(instruction: str, max_results: int = 16) -> list[str]:
    """Optional, incomplete heuristic prompts for the model; never persisted."""
    if not isinstance(instruction, str):
        return []
    candidates = []
    terms = re.compile(
        r"\b(?:max(?:imum)?|min(?:imum)?|limit|must|shall|required|"
        r"at least|at most|exceed|greater|less|under|over|within|"
        r"capacity|tolerance|load|stress|pressure)\b|"
        r"以上|以下|未満|最大|最小|許容|制約|以内",
        re.IGNORECASE,
    )
    for fragment in re.split(r"(?<=[.;。；\n])|\n", instruction[:25000]):
        fragment = fragment.strip()
        if 8 <= len(fragment) <= 420 and NUMBER_PATTERN.search(fragment) and terms.search(fragment):
            candidates.append(fragment[:400])
            if len(candidates) >= max_results:
                break
    return candidates

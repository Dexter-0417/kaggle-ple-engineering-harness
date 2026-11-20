"""Offline evidence-bound, dimensional engineering check tests."""

import asyncio
import json
import pathlib
import tempfile
import time
import unittest

from constraint_check import check_constraint, numeric_constraint_candidates
from serve import Agent


STRESS = "The steel stress shall not exceed 250 MPa under the design load."
SPEC = STRESS + " Feeder current must be no more than 6 A."
DIMLESS = "The total value must be less than or equal to 0.3."


def stress(force="12.5", limit="250", reported=None):
    kwargs = {
        "instruction": SPEC,
        "quote": STRESS,
        "limit_value": limit,
        "limit_unit": "MPa",
        "comparator": "<=",
        "expression": "F/(pi*d**2/4)",
        "variables": {
            "F": {"value": force, "unit": "kN"},
            "d": {"value": "10", "unit": "mm"},
        },
    }
    if reported is not None:
        kwargs["reported_value"] = reported
    return check_constraint(**kwargs)


class ConstraintEngineTests(unittest.TestCase):
    def test_grounded_stress_pass(self):
        x = stress()
        self.assertEqual(x["status"], "PASS")
        self.assertTrue(x["source_quote_matched"])
        self.assertTrue(x["limit_numeric_token_matched"])
        self.assertAlmostEqual(float(x["formula_value_in_limit_unit"]), 159.15, places=1)
        self.assertIn("unverified", x["caveat"])

    def test_exceeding_stress_fails(self):
        x = stress("25")
        self.assertEqual(x["status"], "FAIL")
        self.assertFalse(x["comparison_met"])

    def test_invented_source_quote_unverified(self):
        x = check_constraint(
            instruction="Maximum 250 MPa.",
            quote="The permitted stress shall be maximum 250 MPa.",
            limit_value="250", limit_unit="MPa", comparator="<=",
            expression="p", variables={"p": {"value": "200", "unit": "MPa"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_fabricated_unit_for_real_threshold_is_unverified(self):
        quote = "The pressure limit is 250 kPa and the load is 2 MPa."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="250",
            limit_unit="MPa", comparator="<=", expression="p",
            variables={"p": {"value": "1", "unit": "MPa"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "limit_unit_not_in_quote")

    def test_source_unit_alias_without_exact_text_is_unverified(self):
        quote = "The limit is 250 megapascal."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="250",
            limit_unit="MPa", comparator="<=", expression="p",
            variables={"p": {"value": "100", "unit": "MPa"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")

    def test_changed_limit_not_in_quote_unverified(self):
        x = stress(limit="300")
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "limit_not_in_quote")

    def test_unverified_wrong_formula_unit(self):
        x = check_constraint(
            instruction=STRESS, quote=STRESS,
            limit_value="250", limit_unit="MPa", comparator="<=",
            expression="F", variables={"F": {"value": "10", "unit": "N"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "dimension_mismatch")

    def test_unverified_incompatible_addition(self):
        x = check_constraint(
            instruction="The design length must not exceed 4 m.",
            quote="The design length must not exceed 4 m.",
            limit_value="4", limit_unit="m", comparator="<=",
            expression="x + t",
            variables={
                "x": {"value": "1", "unit": "m"},
                "t": {"value": "1", "unit": "s"},
            },
        )
        self.assertEqual(x["reason"], "dimension_mismatch")

    def test_length_conversion_inches_from_mm(self):
        quote = "Total travel must not exceed 0.5 in."
        for length, expected in (("12", "PASS"), ("13", "FAIL")):
            with self.subTest(length=length):
                x = check_constraint(
                    instruction=quote, quote=quote,
                    limit_value="0.5", limit_unit="in", comparator="<=",
                    expression="travel",
                    variables={"travel": {"value": length, "unit": "mm"}},
                )
                self.assertEqual(x["status"], expected)

    def test_watt_over_volt_recomputes_ampere(self):
        quote = "Feeder current must be no more than 6 A."
        x = check_constraint(
            instruction=SPEC, quote=quote, limit_value="6", limit_unit="A",
            comparator="<=", expression="power / voltage",
            variables={
                "power": {"value": "1200", "unit": "W"},
                "voltage": {"value": "240", "unit": "V"},
            },
        )
        self.assertEqual(x["status"], "PASS")
        self.assertEqual(x["formula_value_in_limit_unit"], "5")

    def test_reported_value_disagreement_fails(self):
        x = stress(reported="20")
        self.assertEqual(x["status"], "FAIL")
        self.assertEqual(x["reason"], "reported_value_disagrees_with_recomputation")

    def test_reported_value_with_rounding_tolerance_passes(self):
        x = stress(reported="159.154943")
        self.assertEqual(x["status"], "PASS")
        self.assertTrue(x["reported_value_consistent"])

    def test_decimal_arithmetic_not_binary_floating_point(self):
        x = check_constraint(
            instruction=DIMLESS, quote=DIMLESS, limit_value="0.3", limit_unit="1",
            comparator="<=", expression="0.1 + 0.2", variables={},
        )
        self.assertEqual(x["status"], "PASS")
        self.assertEqual(x["formula_value_in_limit_unit"], "0.3")

    def test_sqrt_approximate_at_boundary_unverified(self):
        quote = "The required dimensionless upper limit is 2."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="2", limit_unit="1",
            comparator="<=", expression="sqrt(2)**2", variables={},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "approximation_too_close_to_limit")

    def test_sqrt_dimension_is_compatible(self):
        quote = "The height must be less than 4 m."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="4", limit_unit="m",
            comparator="<", expression="sqrt(area)",
            variables={"area": {"value": "9", "unit": "m2"}},
        )
        self.assertEqual(x["status"], "PASS")

    def test_invalid_sqrt_negative_unverified(self):
        quote = "A safe dimensionless limit is 4."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="4", limit_unit="1",
            comparator="<=", expression="sqrt(-1)", variables={},
        )
        self.assertEqual(x["status"], "UNVERIFIED")

    def test_zero_division_unverified(self):
        quote = "The allowed ratio shall not exceed 3."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="3", limit_unit="1",
            comparator="<=", expression="1/(a-a)",
            variables={"a": {"value": "2", "unit": "1"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")

    def test_disallowed_python_constructs(self):
        quote = "The permitted value must be at most 10."
        for expr in (
            "__import__('os').system('touch /tmp/evil')",
            "(1).__class__", "v[0]", "[i for i in range(10)]",
            "lambda: 2", "open('/etc/passwd').read()", "__builtins__",
        ):
            with self.subTest(expr=expr):
                x = check_constraint(
                    instruction=quote, quote=quote, limit_value="10",
                    limit_unit="1", comparator="<=", expression=expr,
                    variables={"v": {"value": "2", "unit": "1"}},
                )
                self.assertEqual(x["status"], "UNVERIFIED")

    def test_pathological_exponent_is_bounded(self):
        quote = "The maximum number is 9."
        before = time.monotonic()
        for expr in ("9**9**9", "2**100", "(" * 300 + "1" + ")" * 300):
            x = check_constraint(
                instruction=quote, quote=quote, limit_value="9", limit_unit="1",
                comparator="<=", expression=expr, variables={},
            )
            self.assertEqual(x["status"], "UNVERIFIED")
        self.assertLess(time.monotonic() - before, 1.0)

    def test_unknown_units_fail_closed(self):
        quote = "The maximum temperature is 50 °C."
        x = check_constraint(
            instruction=quote, quote=quote,
            limit_value="50", limit_unit="°C", comparator="<=",
            expression="T", variables={"T": {"value": "40", "unit": "°C"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "unsupported_unit")

    def test_quote_normalizes_unicode_whitespace(self):
        instruction = "Minimum    clearance\n  30 mm must be maintained."
        quote = "Minimum clearance 30 mm must be maintained."
        x = check_constraint(
            instruction=instruction, quote=quote, limit_value="30",
            limit_unit="mm", comparator=">=", expression="d",
            variables={"d": {"value": "40", "unit": "mm"}},
        )
        self.assertEqual(x["status"], "PASS")

    def test_scientific_notation_limit_matches(self):
        quote = "The current shall not exceed 2e-3 A."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="0.002",
            limit_unit="A", comparator="<=", expression="current",
            variables={"current": {"value": "1", "unit": "mA"}},
        )
        self.assertEqual(x["status"], "PASS")

    def test_token_boundary_blocks_subset_of_large_limit(self):
        instruction = "Minimum clearance 1250 mm must be maintained."
        quote = "clearance 125"
        x = check_constraint(
            instruction=instruction, quote=quote, limit_value="125",
            limit_unit="mm", comparator=">=", expression="d",
            variables={"d": {"value": "200", "unit": "mm"}},
        )
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_token_boundary_blocks_decimal_fragment(self):
        instruction = "Stress shall not exceed 0.250 MPa."
        quote = "250 MPa."
        x = check_constraint(
            instruction=instruction, quote=quote, limit_value="250",
            limit_unit="MPa", comparator="<=", expression="p",
            variables={"p": {"value": "200", "unit": "MPa"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_normalized_quote_too_short_fails(self):
        x = check_constraint(
            instruction="See section 5 for details.",
            quote="       5", limit_value="5", limit_unit="1",
            comparator="<=", expression="4", variables={},
        )
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_decimal_comma_does_not_become_five(self):
        instruction = "Maximum deflection is 0,5 mm."
        x = check_constraint(
            instruction=instruction, quote=instruction, limit_value="5",
            limit_unit="mm", comparator="<=", expression="d",
            variables={"d": {"value": "4", "unit": "mm"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "ambiguous_number_format")

    def test_rounded_division_cannot_false_pass_strict_boundary(self):
        instruction = "The utilisation ratio must be less than 1."
        x = check_constraint(
            instruction=instruction, quote=instruction, limit_value="1",
            limit_unit="1", comparator="<", expression="a/b*b",
            variables={
                "a": {"value": "1", "unit": "1"},
                "b": {"value": "3", "unit": "1"},
            },
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "approximation_too_close_to_limit")

    def test_rounded_division_cannot_false_fail_boundary(self):
        instruction = "The safety ratio must be at least 1."
        x = check_constraint(
            instruction=instruction, quote=instruction, limit_value="1",
            limit_unit="1", comparator=">=", expression="a/b*b",
            variables={
                "a": {"value": "1", "unit": "1"},
                "b": {"value": "3", "unit": "1"},
            },
        )
        self.assertEqual(x["status"], "UNVERIFIED")

    def test_normal_report_value_rounding_is_not_a_failure(self):
        x = stress(reported="159.15")
        self.assertEqual(x["status"], "PASS")
        self.assertTrue(x["reported_value_consistent"])

    def test_integer_derived_result_is_not_scientific_notation(self):
        instruction = "The stress shall not exceed 250 MPa."
        x = check_constraint(
            instruction=instruction, quote=instruction, limit_value="250",
            limit_unit="MPa", comparator="<=", expression="p",
            variables={"p": {"value": "200", "unit": "MPa"}},
        )
        self.assertEqual(x["formula_value_in_limit_unit"], "200")
        self.assertEqual(x["status"], "PASS")

    def test_limit_adjacent_to_unit_matches(self):
        quote = "The stress must not exceed 250MPa."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="250",
            limit_unit="MPa", comparator="<=", expression="p",
            variables={"p": {"value": "200", "unit": "MPa"}},
        )
        self.assertEqual(x["status"], "PASS")

    def test_case_sensitive_quote_prevents_megawatt_milliwatt_mix(self):
        src = "Power shall not exceed 50 mW."
        changed_quote = "Power shall not exceed 50 MW."
        x = check_constraint(
            instruction=src, quote=changed_quote, limit_value="50",
            limit_unit="MW", comparator="<=", expression="p",
            variables={"p": {"value": "10", "unit": "MW"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_case_sensitive_unit_binding_blocks_milli_mismatch(self):
        src = "Power shall not exceed 50 mW."
        x = check_constraint(
            instruction=src, quote=src, limit_value="50",
            limit_unit="MW", comparator="<=", expression="p",
            variables={"p": {"value": "10", "unit": "MW"}},
        )
        self.assertEqual(x["reason"], "limit_unit_not_in_quote")

    def test_simple_unit_cannot_replace_compound_speed(self):
        src = "The velocity shall not exceed 30 m/s."
        x = check_constraint(
            instruction=src, quote=src, limit_value="30",
            limit_unit="m", comparator="<=", expression="x",
            variables={"x": {"value": "20", "unit": "m"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")

    def test_simple_force_unit_cannot_replace_stress_unit(self):
        src = "The pressure shall not exceed 250 N/mm2."
        x = check_constraint(
            instruction=src, quote=src, limit_value="250",
            limit_unit="N", comparator="<=", expression="x",
            variables={"x": {"value": "20", "unit": "N"}},
        )
        self.assertEqual(x["reason"], "limit_unit_not_in_quote")

    def test_volume_flow_written_per_hour_not_volume(self):
        src = "Fluid flow shall not exceed 5 m3 per hour."
        x = check_constraint(
            instruction=src, quote=src, limit_value="5",
            limit_unit="m3", comparator="<=", expression="volume",
            variables={"volume": {"value": "2", "unit": "m3"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")

    def test_dimensionless_cannot_replace_pressure_unit(self):
        src = "Stress shall not exceed 250 MPa."
        x = check_constraint(
            instruction=src, quote=src, limit_value="250",
            limit_unit="1", comparator="<=", expression="x",
            variables={"x": {"value": "20", "unit": "1"}},
        )
        self.assertEqual(x["reason"], "limit_unit_not_in_quote")

    def test_comma_started_numeric_fragment_is_not_a_quote(self):
        src = "The pressure is 1,250 MPa required."
        x = check_constraint(
            instruction=src, quote=",250 MPa required",
            limit_value="250", limit_unit="MPa", comparator="<=",
            expression="p", variables={"p": {"value": "200", "unit": "MPa"}},
        )
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_decimal_dot_started_numeric_fragment_is_not_a_quote(self):
        src = "The displacement is 2.5 mm required."
        x = check_constraint(
            instruction=src, quote=".5 mm required",
            limit_value="0.5", limit_unit="mm", comparator="<=",
            expression="x", variables={"x": {"value": "0.4", "unit": "mm"}},
        )
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_decimal_comma_fragment_is_not_a_quote(self):
        src = "The displacement is 2,5 mm required."
        x = check_constraint(
            instruction=src, quote=",5 mm required",
            limit_value="5", limit_unit="mm", comparator="<=",
            expression="x", variables={"x": {"value": "4", "unit": "mm"}},
        )
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_quote_truncated_just_before_decimal_digit_is_rejected(self):
        src = "The factor of safety 1.5 must be maintained."
        x = check_constraint(
            instruction=src, quote="The factor of safety 1.",
            limit_value="1", limit_unit="1", comparator=">=",
            expression="x", variables={"x": {"value": "1.2", "unit": "1"}},
        )
        self.assertEqual(x["reason"], "quote_not_in_instruction")

    def test_numeric_hyphenated_range_is_ambiguous(self):
        src = "Acceptable clearance is 10-50 mm for all locations."
        x = check_constraint(
            instruction=src, quote=src, limit_value="50",
            limit_unit="mm", comparator="<=", expression="x",
            variables={"x": {"value": "20", "unit": "mm"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "ambiguous_numeric_range_or_grouping")

    def test_grouped_numbers_with_space_are_ambiguous(self):
        src = "The maximum clearance is 1 250 mm."
        x = check_constraint(
            instruction=src, quote=src, limit_value="250",
            limit_unit="mm", comparator="<=", expression="x",
            variables={"x": {"value": "200", "unit": "mm"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")

    def test_catastrophic_cancellation_does_not_false_pass(self):
        src = "The residual shall remain less than 0."
        x = check_constraint(
            instruction=src, quote=src, limit_value="0",
            limit_unit="1", comparator="<",
            expression="a/b*b-a",
            variables={
                "a": {"value": "1e30", "unit": "1"},
                "b": {"value": "3", "unit": "1"},
            },
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "approximation_too_close_to_limit")

    def test_exact_boundary_after_approximation_is_not_contaminated(self):
        check_constraint(
            instruction="Ratio must be less than 1.",
            quote="Ratio must be less than 1.", limit_value="1",
            limit_unit="1", comparator="<", expression="a/b*b",
            variables={"a":{"value":"1","unit":"1"},
                       "b":{"value":"3","unit":"1"}},
        )
        x = check_constraint(
            instruction="The exact maximum limit is 250.",
            quote="The exact maximum limit is 250.", limit_value="250",
            limit_unit="1", comparator="<=", expression="x",
            variables={"x": {"value":"250","unit":"1"}},
        )
        self.assertEqual(x["status"], "PASS")
        self.assertFalse(x["formula_uses_approximation"])

    def test_scientific_reported_coarse_rounding_fails(self):
        src = "The maximum is 150 kPa."
        x = check_constraint(
            instruction=src, quote=src, limit_value="150",
            limit_unit="kPa", comparator="<=", expression="p",
            variables={"p": {"value": "140", "unit": "kPa"}},
            reported_value="1e2",
        )
        self.assertEqual(x["reported_value_consistent"], False)
        self.assertEqual(x["status"], "FAIL")

    def test_obvious_comparator_reversal_is_unverified(self):
        quote = "The steel stress shall not exceed 250 MPa."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="250",
            limit_unit="MPa", comparator=">=",
            expression="p", variables={"p": {"value":"300","unit":"MPa"}},
        )
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "comparator_conflicts_with_quote")

    def test_minimum_reversed_to_maximum_is_unverified(self):
        quote = "Minimum clearance must be at least 30 mm."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="30",
            limit_unit="mm", comparator="<=",
            expression="d", variables={"d": {"value":"20","unit":"mm"}},
        )
        self.assertEqual(x["reason"], "comparator_conflicts_with_quote")

    def test_exact_value_with_clear_direction_is_checked(self):
        quote = "The dimensionless ratio must be exactly 2."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="2",
            limit_unit="1", comparator="==",
            expression="r", variables={"r":{"value":"2","unit":"1"}},
        )
        self.assertEqual(x["status"], "PASS")
        self.assertTrue(x["comparison_direction_checked"])

    def test_unknown_comparator_remains_visibly_unchecked(self):
        quote = "The engineering design value is 2."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="2",
            limit_unit="1", comparator="<=",
            expression="r", variables={"r":{"value":"1","unit":"1"}},
        )
        self.assertFalse(x["comparison_direction_checked"])
        self.assertEqual(x["status"], "UNVERIFIED")
        self.assertEqual(x["reason"], "comparison_direction_not_in_source")

    def test_ambiguous_threshold_never_claims_pass_or_fail(self):
        quote = "The engineering design value is 2."
        for comparator, measured in (("<=", "1"), (">=", "3"), ("==", "2")):
            with self.subTest(comparator=comparator):
                x = check_constraint(
                    instruction=quote, quote=quote,
                    limit_value="2", limit_unit="1",
                    comparator=comparator, expression="r",
                    variables={"r": {"value": measured, "unit": "1"}},
                )
                self.assertEqual(x["status"], "UNVERIFIED")
                self.assertFalse(x["comparison_direction_checked"])

    def test_explicit_bound_still_passes(self):
        quote = "The measured length shall not exceed 2 m."
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="2",
            limit_unit="m", comparator="<=", expression="length",
            variables={"length": {"value": "1.5", "unit": "m"}},
        )
        self.assertEqual(x["status"], "PASS")
        self.assertTrue(x["comparison_direction_checked"])

    def test_japanese_less_than_or_equal_with_unit_suffix(self):
        quote = "電流は6 A以下とする。"
        x = check_constraint(
            instruction=quote, quote=quote, limit_value="6",
            limit_unit="A", comparator="<=",
            expression="I", variables={"I":{"value":"5","unit":"A"}},
        )
        self.assertEqual(x["status"], "PASS")
        self.assertTrue(x["comparison_direction_checked"])

    def test_strange_input_returns_unverified_not_exception(self):
        quote = "The minimum threshold is 4 m."
        values = [None, float("inf"), True, object(), "NaN", "1e400"]
        for value in values:
            x = check_constraint(
                instruction=quote, quote=quote, limit_value="4",
                limit_unit="m", comparator=">=", expression="d",
                variables={"d": {"value": value, "unit": "m"}},
            )
            self.assertEqual(x["status"], "UNVERIFIED")

    def test_candidate_extraction_is_heuristic_not_exhaustive(self):
        text = "Stress shall not exceed 250 MPa.\nUnrelated 100\nAt least 12 mm clearance."
        x = numeric_constraint_candidates(text)
        self.assertGreaterEqual(len(x), 2)
        self.assertTrue(any("250" in s for s in x))
        self.assertFalse(any("Unrelated" in s for s in x))


class FakeSession:
    instruction = "Maximum electric power is 1000 W."
    workdir = "/app"
    limits = {}
    mcp_servers = []
    final_message = ""

    def __init__(self):
        self.logs = []
        self.exec_count = 0

    def time_left(self):
        return 3600

    async def log(self, s):
        self.logs.append(s)

    async def exec(self, command, timeout=120):
        self.exec_count += 1
        raise AssertionError("check_constraint must not execute task commands")


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_uses_current_instruction_without_exec(self):
        fake = FakeSession()
        agent = Agent(fake)
        result = await agent.run_tool("check_constraint", {
            "quote": fake.instruction, "limit_value": "1000", "limit_unit": "W",
            "comparator": "<=", "expression": "P",
            "variables": {"P": {"value": "0.8", "unit": "kW"}},
        })
        data = json.loads(result)
        self.assertEqual(data["status"], "PASS")
        self.assertEqual(fake.exec_count, 0)
        self.assertFalse(any(fake.instruction in line for line in fake.logs))

    async def test_tool_does_not_log_hidden_task_prompt(self):
        fake = FakeSession()
        agent = Agent(fake)
        await agent.run_tool("check_constraint", {
            "quote": fake.instruction, "limit_value": "1000", "limit_unit": "W",
            "comparator": "<=", "expression": "P", "variables": {},
        })
        self.assertFalse(any("Maximum electric power" in line for line in fake.logs))

    async def test_missing_required_fields_return_error(self):
        fake = FakeSession()
        result = await Agent(fake).run_tool("check_constraint", {"quote": "hi"})
        self.assertIn("error:", result)

    async def test_unexpected_arg_name_never_leaks_instruction_to_logs(self):
        fake = FakeSession()
        agent = Agent(fake)
        params = {
            "quote": fake.instruction, "limit_value": "1000", "limit_unit": "W",
            "comparator": "<=", "expression": "P",
            "variables": {"P": {"value": "1", "unit": "W"}},
        }
        params[fake.instruction] = "unexpected input name"
        output = json.loads(await agent.run_tool("check_constraint", params))
        self.assertEqual(output["status"], "PASS")
        self.assertFalse(any(fake.instruction in line for line in fake.logs))

    async def test_variables_json_string_parses_safely(self):
        fake = FakeSession()
        result = json.loads(await Agent(fake).run_tool("check_constraint", {
            "quote": fake.instruction, "limit_value": "1000", "limit_unit": "W",
            "comparator": "<=", "expression": "P",
            "variables": json.dumps({"P": {"value": "500", "unit": "W"}}),
        }))
        self.assertEqual(result["status"], "PASS")

    async def test_required_schema_contains_nested_units(self):
        from serve import TOOLS
        check = next(
            t["function"] for t in TOOLS if t["function"]["name"] == "check_constraint"
        )
        var_schema = check["parameters"]["properties"]["variables"]
        self.assertIn("additionalProperties", var_schema)
        self.assertEqual(
            var_schema["additionalProperties"]["required"], ["value", "unit"]
        )

    async def test_candidates_tool_returns_original_snippets(self):
        fake = FakeSession()
        result = json.loads(await Agent(fake).run_tool("suggest_constraints", {}))
        self.assertTrue(any("1000" in x for x in result["candidate_quotes"]))
        self.assertEqual(fake.exec_count, 0)


if __name__ == "__main__":
    unittest.main()

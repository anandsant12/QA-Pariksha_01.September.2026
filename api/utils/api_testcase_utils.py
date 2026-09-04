"""
api/utils/api_testcase_utils.py

Generates positive/negative API test cases from a single-API JSON spec
(api_name, api_url, method, payload{field: {value, required, validation}}).

Design constraints:
  - Exactly ONE field is mutated per generated NEGATIVE test case (no permutations).
  - A field's "value" may be a single value OR a list of valid values. Only values
    already present in that list may ever be used in a POSITIVE test case — the LLM
    is never allowed to invent a new "correct" value.
  - 4 mandatory fields are auto-injected / validated before the LLM ever sees the
    payload: REQUEST_AUTH_ID, REQUEST_TELLER_ID, BRANCH_CODE (auto-filled with
    defaults if missing) and SOURCE_ID (hard error if missing). REQUEST_REFERENCE_NUMBER
    (RRN) is derived from SOURCE_ID unless already supplied.
"""
import json
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional
from api.utils.azure_utility import client, MODEL


# ============================================================================
# Mandatory field handling
# ============================================================================
MANDATORY_DEFAULT_FIELDS: Dict[str, Dict[str, Any]] = {
    "REQUEST_AUTH_ID": {
        "value": "1036662",
        "required": True,
        "validation": "type should be string, maxlength should be 7",
    },
    "REQUEST_TELLER_ID": {
        "value": "1015421",
        "required": True,
        "validation": "type should be string, maxlength should be 7",
    },
    "BRANCH_CODE": {
        "value": "00437",
        "required": True,
        "validation": "type should be string, maxlength should be 5",
    },
}
SOURCE_ID_FIELD = "SOURCE_ID"
RRN_FIELD = "REQUEST_REFERENCE_NUMBER"

# ── EIS Channel / EIS Microservices: parent-wrapped payload shape ──────────
# { SOURCE_ID, DESTINATION, TXN_TYPE, TXN_SUB_TYPE, REQUEST_REFERENCE_NUMBER?, EIS_PAYLOAD }
# Only EIS_PAYLOAD's inner fields are ever exercised by the LLM; the four other
# parent fields stay fixed at their baseline value in every generated test case,
# and RRN is freshly generated per test case regardless of whether the user
# supplied one (a banking RRN must be unique per call).
DESTINATION_FIELD    = "DESTINATION"
TXN_TYPE_FIELD       = "TXN_TYPE"
TXN_SUB_TYPE_FIELD   = "TXN_SUB_TYPE"
EIS_PAYLOAD_FIELD    = "EIS_PAYLOAD"
WRAPPED_PARENT_FIELDS = [SOURCE_ID_FIELD, DESTINATION_FIELD, TXN_TYPE_FIELD, TXN_SUB_TYPE_FIELD, EIS_PAYLOAD_FIELD]


RRN_TOTAL_LENGTH = 25
RRN_PREFIX = "SBI"

# Running sequence number (NNNNNN) — process-local, thread-safe, wraps at 999999.
_rrn_sequence_lock = threading.Lock()
_rrn_sequence_counter = 0
_RRN_SEQUENCE_MULTIPLIER = 400223  # must stay coprime with 1_000_000 (not divisible by 2 or 5)


def _next_sequence_number() -> str:
    """Thread-safe running sequence number, 6 digits. Collision-free across a
    full 1,000,000-call cycle (same guarantee as a plain counter), but scrambled
    so consecutive calls don't look sequential."""
    global _rrn_sequence_counter
    with _rrn_sequence_lock:
        _rrn_sequence_counter = (_rrn_sequence_counter + 1) % 1_000_000
        counter = _rrn_sequence_counter
    scrambled = (counter * _RRN_SEQUENCE_MULTIPLIER) % 1_000_000
    return f"{scrambled:06d}"


def make_rrn(source_id: str) -> str:
    """
    Builds a REQUEST_REFERENCE_NUMBER in the bank-mandated format:

        SBI   XX   YYDDD   HHmmssSSS   NNNNNN
        (3)  (2)    (5)       (9)        (6)     = 25 characters total

    - SBI       : fixed literal prefix.
    - XX        : Channel Identifier, taken from SOURCE_ID (must be exactly 2 chars,
                  e.g. "LT" for the YONO channel).
    - YYDDD     : Julian date — 2-digit year + 3-digit day-of-year
                  (e.g. 26-Feb-2020 -> "20" + "057" = "20057").
    - HHmmssSSS : current time — hour, minute, second, millisecond.
    - NNNNNN    : 6-digit running sequence number (wraps at 999999).
    """
    channel_id = str(source_id).strip().upper()
    if len(channel_id) != 2:
        raise ValueError(
            f"SOURCE_ID '{source_id}' must be exactly 2 characters to be used as the "
            f"RRN Channel Identifier (XX), got {len(channel_id)} characters."
        )

    now = datetime.now(ZoneInfo("Asia/Kolkata"))  # bank-local (IST) timestamp

    julian_date = f"{now.strftime('%y')}{now.timetuple().tm_yday:03d}"        # YYDDD
    time_part   = f"{now.strftime('%H%M%S')}{now.microsecond // 1000:03d}"    # HHmmssSSS
    sequence    = _next_sequence_number()                                     # NNNNNN

    rrn = f"{RRN_PREFIX}{channel_id}{julian_date}{time_part}{sequence}"

    assert len(rrn) == RRN_TOTAL_LENGTH, f"Generated RRN is {len(rrn)} chars, expected {RRN_TOTAL_LENGTH}"
    return rrn


def _field_values(field_spec: Dict[str, Any]) -> List[Any]:
    """A field's 'value' may be a scalar or a list — always return it as a list."""
    v = field_spec.get("value")
    if isinstance(v, list):
        return v if v else [None]
    return [v]


def _baseline_value(field_spec: Dict[str, Any]) -> Any:
    """The first value is always treated as the baseline/default correct value."""
    return _field_values(field_spec)[0]


def apply_mandatory_fields(
    payload: Dict[str, Dict[str, Any]],
    include_fields: Optional[List[str]] = None,
) -> tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """
    - Auto-injects REQUEST_AUTH_ID / REQUEST_TELLER_ID / BRANCH_CODE with defaults
      if any of them is missing from the input payload — but ONLY for the field
      names present in `include_fields`. Each checked independently.
    - `include_fields=None` (default) means "inject all three if missing" — this
      matches the UI checkboxes being ticked ON by default. Pass an explicit list
      (e.g. from the UI's unticked checkboxes) to skip specific fields entirely —
      an unticked field will NOT be added even if missing from the payload.
    ...
    Returns: (payload_with_mandatory_fields, generated_rrn_or_None)
    """
    payload = dict(payload)  # shallow copy — don't mutate the caller's dict

    fields_to_inject = (
        set(MANDATORY_DEFAULT_FIELDS.keys()) if include_fields is None
        else set(include_fields) & set(MANDATORY_DEFAULT_FIELDS.keys())
    )

    for field_name in fields_to_inject:
        if field_name not in payload:
            payload[field_name] = dict(MANDATORY_DEFAULT_FIELDS[field_name])

    source_spec = payload.get(SOURCE_ID_FIELD)
    if not source_spec or _baseline_value(source_spec) in (None, ""):
        raise ValueError(
            f"'{SOURCE_ID_FIELD}' is mandatory. Please provide {SOURCE_ID_FIELD} in the input payload."
        )

    rrn_spec = payload.get(RRN_FIELD)
    rrn_already_provided = bool(rrn_spec) and _baseline_value(rrn_spec) not in (None, "")

    generated_rrn: Optional[str] = None
    if not rrn_already_provided:
        source_id_value = str(_baseline_value(source_spec))
        generated_rrn = make_rrn(source_id_value)
        # Intentionally NOT added to `payload` — it must not appear in Test Data
        # unless the user supplied it themselves.

    return payload, generated_rrn


# ============================================================================
# EIS Channel / EIS Microservices — parent-wrapped payload handling
# ============================================================================

def _find_key_ci(d: Dict[str, Any], name: str) -> Optional[str]:
    """Case-insensitive lookup of a key name inside dict d — returns the actual
    key as it appears in d, or None if not present."""
    for k in d.keys():
        if isinstance(k, str) and k.lower() == name.lower():
            return k
    return None


def split_eis_payload_container(eis_payload_value: Any) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str], str]:
    """
    EIS_PAYLOAD's "value" can be either:
      (a) FLAT   — a dict of field_name -> {value, required, validation} to test
                    directly, e.g. {"mobile_number": {...}, "pan_number": {...}}.
      (b) NESTED — a dict with a "HEADERS" key (static request headers, sent as-is,
                    NEVER varied/tested) and a "BODY" key (dict of field_name ->
                    {value, required, validation} to test) — used when the target
                    downstream endpoint expects headers alongside a body.

    Both shapes are valid under EITHER api_type (EIS Channel or EIS Microservices)
    — the shape is a property of what the specific downstream endpoint expects,
    NOT of which api_type was selected, so this is detected purely from the JSON
    structure and never inferred from api_type.

    Returns (testable_fields, headers_or_None, headers_key_or_None, body_key_used).
    `body_key_used` is "value" for the flat shape (there's no real "BODY" key to
    report — it's a placeholder so callers always get a string back).
    """
    if not isinstance(eis_payload_value, dict) or not eis_payload_value:
        raise ValueError(
            "EIS_PAYLOAD.value must be an object — either field specs directly, or "
            "a {HEADERS, BODY} object with BODY holding the field specs."
        )

    body_key = _find_key_ci(eis_payload_value, "BODY")
    if body_key is not None:
        headers_key = _find_key_ci(eis_payload_value, "HEADERS")
        headers = eis_payload_value.get(headers_key) if headers_key else {}
        body = eis_payload_value.get(body_key)
        if not isinstance(body, dict) or not body:
            raise ValueError(f"EIS_PAYLOAD.value.{body_key} must be a non-empty object of field specs.")
        return body, (headers if isinstance(headers, dict) else {}), (headers_key or "HEADERS"), body_key

    # Flat shape — the whole object is field specs.
    return eis_payload_value, None, None, "value"


def apply_wrapped_mandatory_fields(
    payload: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]], Optional[str], str, str]:
    """
    Validates and unpacks the parent-wrapped payload shape used by the EIS Channel
    and EIS Microservices API types:

        { SOURCE_ID, DESTINATION, TXN_TYPE, TXN_SUB_TYPE, REQUEST_REFERENCE_NUMBER?, EIS_PAYLOAD }

    SOURCE_ID / DESTINATION / TXN_TYPE / TXN_SUB_TYPE / EIS_PAYLOAD are mandatory.
    REQUEST_REFERENCE_NUMBER is never required here — a fresh, unique one is
    generated for every test case at wrap time regardless of whether the user
    supplied one (see wrap_testcases_in_eis_container) — a banking RRN must be
    unique per call, so reusing a single supplied value across every test case
    would just get every row rejected as a duplicate once actually run.

    Returns:
      (payload, testable_fields, headers_or_None, headers_key_or_None, body_key_used, source_id_value)
    """
    payload = dict(payload)  # shallow copy — don't mutate the caller's dict

    missing = [f for f in WRAPPED_PARENT_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"Missing mandatory field(s) for this API type: {', '.join(missing)}.")

    for field_name in (SOURCE_ID_FIELD, DESTINATION_FIELD, TXN_TYPE_FIELD, TXN_SUB_TYPE_FIELD):
        if _baseline_value(payload[field_name]) in (None, ""):
            raise ValueError(f"'{field_name}' is mandatory. Please provide a value for it.")

    source_id_value = str(_baseline_value(payload[SOURCE_ID_FIELD])).strip().upper()
    if len(source_id_value) != 2:
        raise ValueError(
            f"SOURCE_ID '{source_id_value}' must be exactly 2 characters for this API type — "
            f"it's used as the RRN channel identifier for every generated test case."
        )

    eis_spec = payload[EIS_PAYLOAD_FIELD]
    eis_value = eis_spec.get("value") if isinstance(eis_spec, dict) else None
    testable_fields, headers, headers_key, body_key = split_eis_payload_container(eis_value)
    if not testable_fields:
        raise ValueError("EIS_PAYLOAD must contain at least one field to test.")

    return payload, testable_fields, headers, headers_key, body_key, source_id_value


def wrap_testcases_in_eis_container(
    testcases: List[Dict[str, Any]],
    payload: Dict[str, Dict[str, Any]],
    headers: Optional[Dict[str, Any]],
    headers_key: Optional[str],
    body_key: str,
    source_id_value: str,
) -> List[Dict[str, Any]]:
    """
    Rebuilds every generated test case's "Test Data" into the full envelope the EIS
    Channel / EIS Microservices API types expect: SOURCE_ID / DESTINATION / TXN_TYPE
    / TXN_SUB_TYPE at their fixed baseline values (unchanged across every test case
    — only EIS_PAYLOAD is ever varied), a freshly generated unique
    REQUEST_REFERENCE_NUMBER per test case, and EIS_PAYLOAD itself rebuilt in
    whichever shape (flat, or {HEADERS, BODY}) the input used.

    This REQUEST_REFERENCE_NUMBER is what run_api_testcases_endpoint later passes
    to call_eis_api as the `rrn` argument — since it lives right here inside the
    SAME dict that gets encrypted, and call_eis_api also places its `rrn` argument
    into the outer unencrypted envelope, the two always match automatically.
    """
    parent_baseline = {
        SOURCE_ID_FIELD:    _baseline_value(payload[SOURCE_ID_FIELD]),
        DESTINATION_FIELD:  _baseline_value(payload[DESTINATION_FIELD]),
        TXN_TYPE_FIELD:     _baseline_value(payload[TXN_TYPE_FIELD]),
        TXN_SUB_TYPE_FIELD: _baseline_value(payload[TXN_SUB_TYPE_FIELD]),
    }

    wrapped: List[Dict[str, Any]] = []
    for tc in testcases:
        inner_td = tc.get("Test Data") or {}
        eis_payload_value: Any = {headers_key: (headers or {}), body_key: inner_td} if headers_key else inner_td

        full_td = {
            **parent_baseline,
            RRN_FIELD: make_rrn(source_id_value),
            EIS_PAYLOAD_FIELD: eis_payload_value,
        }
        wrapped.append({**tc, "Test Data": full_td})
    return wrapped


def build_eis_payload_baseline(
    testable_fields: Dict[str, Dict[str, Any]],
    headers: Optional[Dict[str, Any]],
    headers_key: Optional[str],
    body_key: str,
) -> Any:
    """The "true" EIS_PAYLOAD value exactly as given in the input JSON — every
    field at its baseline (first/default) value, rebuilt in whichever shape
    (flat, or {HEADERS, BODY}) the input used. Used to hold EIS_PAYLOAD fixed
    while parent-key test cases vary SOURCE_ID / DESTINATION / TXN_TYPE /
    TXN_SUB_TYPE instead."""
    baseline_fields = {f: _baseline_value(spec) for f, spec in testable_fields.items()}
    if headers_key is not None:
        return {headers_key: (headers or {}), body_key: baseline_fields}
    return baseline_fields


def _is_full_baseline(test_data: Any, fields: Dict[str, Dict[str, Any]]) -> bool:
    """True if every field in test_data matches its baseline (first/default)
    value — i.e. this is the "everything correct" case. Used to drop the
    parent-field batch's own baseline-positive test case, since that exact
    scenario (everything at baseline, including EIS_PAYLOAD) is already
    produced once by the EIS_PAYLOAD-variation batch — no need to duplicate it."""
    if not isinstance(test_data, dict):
        return False
    for field, spec in fields.items():
        if field not in test_data:
            return False
        if str(test_data.get(field)) != str(_baseline_value(spec)):
            return False
    return True


def wrap_parent_field_testcases(
    testcases: List[Dict[str, Any]],
    eis_payload_baseline: Any,
    source_id_for_rrn: str,
) -> List[Dict[str, Any]]:
    """
    Rebuilds every generated "parent field" test case's Test Data: whichever
    subset of {SOURCE_ID, DESTINATION, TXN_TYPE, TXN_SUB_TYPE} the LLM produced
    for THIS test case (exactly one deliberately varied or removed, per the
    usual one-field-at-a-time rule) is used as-is, EIS_PAYLOAD is fixed at its
    baseline ("true") value from the input JSON, and every test case gets a
    fresh unique REQUEST_REFERENCE_NUMBER generated from the ORIGINAL valid
    SOURCE_ID — not whatever value is under test in that row, since a
    deliberately missing/malformed SOURCE_ID test case still needs a
    well-formed RRN for the call to actually reach the API.
    """
    wrapped: List[Dict[str, Any]] = []
    for tc in testcases:
        parent_td = dict(tc.get("Test Data") or {})
        full_td = {
            **parent_td,
            RRN_FIELD: make_rrn(source_id_for_rrn),
            EIS_PAYLOAD_FIELD: eis_payload_baseline,
        }
        wrapped.append({**tc, "Test Data": full_td})
    return wrapped


API_TESTCASE_SYSTEM_PROMPT = """You are an expert API test analyst for a banking application (State Bank of India).
You generate POSITIVE and NEGATIVE test cases for a SINGLE REST API from a payload
specification that gives, for every field: its correct value(s), whether it is
required, and a plain-English validation rule (may be empty).

## VALUE FORMAT
Each field's "value" is EITHER:
  - a single value (e.g. "value": "9876543210"), OR
  - a LIST of multiple values (e.g. "value": ["9876543210", "9123456780"])
When it is a list, EVERY entry in that list is an independently CORRECT / VALID value
for that field — not a range, not an example to interpolate from, just a fixed set of
acceptable values.

## STRICT RULES — FOLLOW EXACTLY

1. BASELINE POSITIVE TEST CASE
   - Always create exactly ONE positive test case using the FIRST value of every
     field's "value" (if it's a list, use value[0]; if scalar, use it as-is).
   - Description example: "To verify <api_name> with correct payload fetches correct
     details from the API."

2. NEVER INVENT A "CORRECT" VALUE — CRITICAL
   - For POSITIVE test cases, you may ONLY use a value for a field IF that exact value
     already appears in that field's given "value" list (or is its scalar value).
   - You must NEVER invent, guess, or construct a new value that you believe would
     also be valid — even if it seems like it satisfies the stated validation rule.
   - If a field's "value" is a list with MORE THAN ONE entry, you MAY create one
     additional POSITIVE test case per EXTRA entry in that list (varying ONLY that
     field, all other fields at their baseline/first value). Description: "To verify
     <api_name> accepts a valid <field> (<value>) and processes the request
     successfully."
   - If a field's "value" is a single scalar (or a list with only ONE entry), do NOT
     create any additional positive variant for that field — only the baseline covers
     it.
   - This restriction applies ONLY to POSITIVE test cases. Negative test cases are
     expected to use invalid/broken values by design (see rule 4).

3. VARY ONLY ONE FIELD AT A TIME — NEVER COMBINE
   - Every additional test case (positive variant or negative) changes ONLY ONE field
     from its baseline value. ALL other fields MUST keep their baseline (first) value.
   - Do NOT generate permutations or combinations across multiple fields.

4. NEGATIVE TEST CASES (only for fields with a non-empty "validation")
   - If required=true: exactly ONE negative test case with that field's KEY REMOVED
     from the payload entirely (this tests "mandatory field missing" — do not send an
     empty string for this case, actually omit the key).
   - You can test this even if require field is like required=false means (not required). For this also
     please check with that field's KEY REMOVED
     from the payload entirely
   - Further NEGATIVE test cases, each breaking exactly ONE distinct aspect of
     the stated validation rule (e.g. for "10 digit numeric": too short, too long,
     contains alphabets). Pick the most realistic and relevant ones — do not invent
     rules that were not stated.
   - Fields with an empty/missing "validation" and required=false are skipped
     entirely — do not guess rules for them.

5. FUNCTION DESCRIPTION
   - "Function Description" is the API's name/purpose and must be IDENTICAL across
     every test case in this batch, e.g. "Mobile Number Enquiry API".

6. TEST CASE DESCRIPTION — must state field + rule being verified
   - Positive baseline: "To verify <api_name> with correct payload fetches correct
     details from the API."
   - Positive field variant (only when field has multiple valid values): "To verify
     <api_name> accepts a valid <field> and processes the request successfully."
   - Negative: "To verify <api_name> with <field> <plain description of the
     violation> displays an error."
     e.g. "To verify Mobile Enquiry API with RRN greater than 25 characters displays
     an error."
     e.g. "To verify Mobile Enquiry API with mandatory field mobile_number missing
     displays an error."

7. TEST DATA
   - "Test Data" is the COMPLETE JSON payload object (all fields, using their
     baseline values unless that field is the one under test) for that specific test
     case, with the single deliberate change applied (or no change, for the
     baseline). It must be valid JSON.
       
8. OUTPUT FORMAT — JSON ONLY, NOTHING ELSE
   Return ONLY a JSON array. Each element must have EXACTLY these keys:
     "Function Description": string
     "Test Case Description": string
     "Test Case Type": "Positive" or "Negative"
     "Test Data": object (the full payload for this test case)
   Do NOT include "Test Case ID", "API_URL", or "API_METHOD" — those are added by the
   caller. Do not wrap the array in another object. Do not add markdown fences or
   commentary."""


def _strip_json_fences(content: str) -> str:
    content = (content or "").strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
    return content.replace("```json", "").replace("```", "").strip()


def _sanitize_positive_testcases(
    testcases: List[Dict[str, Any]],
    payload: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Defensive guard for rule #2: if the LLM ever slips and invents a "valid" value
    for a field in a POSITIVE test case that wasn't in the field's given value list,
    force it back to that field's baseline value.
    """
    allowed_values = {
        field: set(str(v) for v in _field_values(spec))
        for field, spec in payload.items()
    }
    baseline = {field: _baseline_value(spec) for field, spec in payload.items()}

    for tc in testcases:
        if tc.get("Test Case Type") != "Positive":
            continue
        td = tc.get("Test Data") or {}
        for field, val in list(td.items()):
            if field in allowed_values and str(val) not in allowed_values[field]:
                td[field] = baseline.get(field, val)
        tc["Test Data"] = td
    return testcases


def _assign_unique_rrns(
    testcases: List[Dict[str, Any]],
    baseline_rrn: str,
    source_id: str,
) -> List[Dict[str, Any]]:
    """
    Whenever the user supplies REQUEST_REFERENCE_NUMBER in the input payload, the
    LLM (correctly, per rule 3 — vary only one field per test case) carries that
    SAME literal value into every test case's Test Data. Left as-is, every row
    would end up with an IDENTICAL RRN, which the target banking API rejects as a
    duplicate/non-unique reference number once test cases are actually executed.

    This gives every test case its OWN unique RRN (same "SBI"+SOURCE_ID prefix,
    fresh random suffix, still 25 chars total) — EXCEPT test cases where the LLM
    deliberately mutated the RRN itself to build a negative test case FOR the RRN
    field (its value no longer matches the original baseline). Those are left
    untouched — overwriting them would silently defeat that specific negative test.
    """
    for tc in testcases:
        td = tc.get("Test Data")
        if not isinstance(td, dict) or RRN_FIELD not in td:
            continue
        current_val = str(td[RRN_FIELD])
        if current_val == str(baseline_rrn):
            # Untouched by the LLM -> safe to give this row its own unique RRN.
            td[RRN_FIELD] = make_rrn(source_id)
        # else: LLM intentionally broke this field for a negative test -> leave as-is.
    return testcases

def generate_api_testcases_via_llm(
    api_name: str,
    api_url: str,
    method: str,
    payload: Dict[str, Any],
    user_prompt: Optional[str] = None,
    testcase_type: str = "UAT",
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)

    user_message = (
        f"Generate positive and negative test cases for the following API.\n\n"
        f"API Name: {api_name}\n"
        f"API URL: {api_url}\n"
        f"HTTP Method: {method.upper()}\n"
        f"Test Environment: {testcase_type}\n\n"
        f"Payload specification (field -> value(s) / required / validation):\n"
        f"{payload_json}\n"
    )
    if user_prompt:
        user_message += f"\nAdditional instructions from the user:\n{user_prompt}\n"
    user_message += (
        "\nFollow all rules from the system prompt strictly. Never invent a 'valid' "
        "value that isn't already listed for a field. Vary ONLY one field per test "
        "case. Also try to cover all the possible scenarios, edge cases as well. Return the JSON array only."
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Generating API test cases (attempt {attempt}/{max_retries})…")
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": API_TESTCASE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=32000,
            )
            content = _strip_json_fences(response.choices[0].message.content)
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("LLM response is not a JSON array")

            cleaned: List[Dict[str, Any]] = []
            for idx, tc in enumerate(parsed, start=1):
                if not isinstance(tc, dict):
                    continue
                td = tc.get("Test Data")
                if isinstance(td, str):
                    try:
                        td = json.loads(td)
                    except Exception:
                        pass
                tc_type = str(tc.get("Test Case Type", "")).strip().capitalize()
                if tc_type not in ("Positive", "Negative"):
                    tc_type = "Negative" if "error" in str(tc.get("Test Case Description", "")).lower() else "Positive"
                cleaned.append({
                    "Function Description": tc.get("Function Description", api_name),
                    "Test Case ID": f"TC_API_{idx:03d}",
                    "Test Case Description": str(tc.get("Test Case Description", "")).strip(),
                    "Test Case Type": tc_type,
                    "API_URL": api_url,
                    "API_METHOD": method.upper(),
                    "Test Data": td if isinstance(td, dict) else {},
                })

            if not cleaned:
                raise ValueError("No valid test cases parsed from LLM response")

            cleaned = _sanitize_positive_testcases(cleaned, payload)
            rrn_spec = payload.get(RRN_FIELD)
            if rrn_spec:
                baseline_rrn = _baseline_value(rrn_spec)
                source_spec  = payload.get(SOURCE_ID_FIELD)
                if baseline_rrn not in (None, "") and source_spec:
                    source_id_value = str(_baseline_value(source_spec))
                    cleaned = _assign_unique_rrns(cleaned, baseline_rrn, source_id_value)

            # Re-number IDs sequentially after sanitization (order unchanged, just clarity)
            for i, tc in enumerate(cleaned, start=1):
                tc["Test Case ID"] = f"TC_API_{i:03d}"

            print(f"    ✓ Generated {len(cleaned)} API test cases")
            return cleaned

        except Exception as e:
            last_error = e
            print(f"    ✗ Attempt {attempt} failed: {e}")

    raise RuntimeError(f"API test case generation failed after {max_retries} attempts: {last_error}")


# ============================================================================
# EIS Channel / EIS Microservices — parent-key variation batch
# ============================================================================

def generate_parent_field_testcases(
    api_name: str,
    api_url: str,
    method: str,
    payload: Dict[str, Dict[str, Any]],
    eis_payload_baseline: Any,
    source_id_for_rrn: str,
    user_prompt: Optional[str] = None,
    testcase_type: str = "UAT",
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generates positive/negative test cases that vary ONE parent field at a time
    — SOURCE_ID / DESTINATION / TXN_TYPE / TXN_SUB_TYPE — while EIS_PAYLOAD is
    held fixed at its baseline ("true") value from the input JSON. All four
    parent fields are treated as mandatory regardless of what their "required"
    flag says in the input, since they're structurally compulsory for this API
    type. Reuses the exact same LLM engine as EIS_PAYLOAD field generation —
    it's simply pointed at the parent fields instead of EIS_PAYLOAD's fields.

    The pure "everything at baseline" case is stripped from the result (it's
    already produced once, in the EIS_PAYLOAD-variation batch), and every
    remaining case is wrapped with the fixed EIS_PAYLOAD baseline plus a
    freshly generated, unique RRN.
    """
    parent_fields = {
        f: {**payload[f], "required": True}  # structurally mandatory for this API type
        for f in (SOURCE_ID_FIELD, DESTINATION_FIELD, TXN_TYPE_FIELD, TXN_SUB_TYPE_FIELD)
    }

    testcases = generate_api_testcases_via_llm(
        api_name=api_name,
        api_url=api_url,
        method=method,
        payload=parent_fields,
        user_prompt=user_prompt,
        testcase_type=testcase_type,
        max_retries=max_retries,
    )

    testcases = [
        tc for tc in testcases
        if not (tc.get("Test Case Type") == "Positive" and _is_full_baseline(tc.get("Test Data"), parent_fields))
    ]

    return wrap_parent_field_testcases(testcases, eis_payload_baseline, source_id_for_rrn)



# ============================================================================
# Pass/Fail evaluation (post Run-Automation)
# ============================================================================
EVALUATE_TESTCASE_SYSTEM_PROMPT = """You are an expert API test analyst for a banking application (State Bank of India).
You are given a batch of already-executed API test cases. For each one you receive:
  - "Test Case Description": what the test case is supposed to verify
  - "Test Case Type": "Positive" or "Negative"
  - "Actual_Response": the real response text captured after actually calling the API

Decide whether each test case PASSED or FAILED, based ONLY on whether the actual
response matches what the description says should happen for that test type.

## READING THE ACTUAL_RESPONSE — TWO POSSIBLE SHAPES

Actual_Response is either a plain-text gateway/network error, or decrypted JSON
in ONE of these two shapes — figure out which one you're looking at first:

1. FLAT — the business result is directly in the top-level fields:
   RESPONSE_STATUS ("0" = accepted/success, "1" or "2" = rejected), ERROR_CODE /
   ERROR_DESCRIPTION (non-empty = rejected), or the whole response is a
   "GATEWAY ERROR [...]" string (always a rejection). This is what plain EIS
   calls return, and also what a gateway-level rejection looks like for any
   call type.

2. NESTED under "EIS_RESPONSE" — this is what EIS Channel / EIS Microservices
   calls return: the DOWNSTREAM service's own response sits inside an
   "EIS_RESPONSE" object, with its OWN "success" (true/false), "statusCode"
   (HTTP-style: 2xx = success, 4xx/5xx = rejection), and "errors" (non-null /
   non-empty = rejected) fields. Top-level RESPONSE_STATUS / ERROR_CODE /
   ERROR_DESCRIPTION sitting ALONGSIDE "EIS_RESPONSE" only report whether the
   EIS gateway itself successfully routed the request and got a reply — that
   is a SEPARATE concern from whether the downstream service accepted the
   payload, and it is very common (and NOT a bug) for RESPONSE_STATUS to be
   "0" (gateway worked fine) at the very same time "EIS_RESPONSE" shows
   "success": false / a 4xx-5xx "statusCode" / non-empty "errors" (the
   downstream service rejected the input). Whenever "EIS_RESPONSE" is present,
   you MUST judge PASS/FAIL from ITS fields, NOT from the top-level
   RESPONSE_STATUS / ERROR_CODE / ERROR_DESCRIPTION next to it. The exact field
   names inside the nested envelope may vary slightly by downstream service
   (e.g. "status" instead of "statusCode", errors as a list instead of an
   object) — use judgment on the equivalent business-outcome signal, but the
   core principle always holds: a nested envelope's OWN outcome wins over the
   outer gateway-level fields sitting alongside it.

## RULES

1. POSITIVE test cases PASS when the business result (per whichever shape
   above applies) indicates SUCCESS — e.g. flat RESPONSE_STATUS "0" with no
   ERROR_CODE/ERROR_DESCRIPTION, or (when EIS_RESPONSE is present)
   EIS_RESPONSE.success is true / statusCode is 2xx / errors is null or empty,
   with correct customer/business data returned.
   They FAIL when the business result shows an error/rejection instead of
   success.

2. NEGATIVE test cases PASS when the business result (per whichever shape
   above applies) indicates the API correctly REJECTED the invalid input —
   e.g. flat RESPONSE_STATUS "1"/"2" or a non-empty ERROR_CODE/ERROR_DESCRIPTION,
   any "GATEWAY ERROR [...]" message, or (when EIS_RESPONSE is present)
   EIS_RESPONSE.success is false / statusCode is 4xx-5xx / errors is non-empty.
   They FAIL when the business result indicates the API incorrectly ACCEPTED
   the bad input and returned success/business data anyway (this is a real
   bug — the validation that should have blocked it did not).

3. If the Actual_Response is empty, or begins with "ERROR:" describing a network,
   timeout, decryption, or key-loading failure (NOT a business/gateway rejection),
   mark it FAILED with a Remarks note that execution itself did not complete
   (this is inconclusive, not a validation success or failure — but for pass/fail
   purposes it must be treated as FAILED so it gets flagged for manual review).

4. Judge each test case independently. Do not let one test case's error code
   influence your judgment of another.

5. Be objective — do not assume test intent beyond what "Test Case Description"
   and "Test Case Type" literally state. If the actual response is ambiguous
   relative to the description, prefer FAILED with a Remarks note explaining the
   ambiguity, rather than guessing PASS.

## OUTPUT FORMAT — JSON ONLY, NOTHING ELSE
Return ONLY a JSON array, one element per input test case, in the SAME ORDER as
given, with EXACTLY these keys:
  "Test Case ID": string (copy from input, unchanged)
  "Pass_Fail": "Pass" or "Fail"
  "Remarks": string (ONE short sentence explaining the verdict, referencing what
             the actual response actually showed)
Do not add markdown fences or commentary. Do not omit any test case."""


def evaluate_api_testcases_via_llm(
    testcases: List[Dict[str, Any]],
    max_retries: int = 3,
) -> Dict[str, Dict[str, str]]:
    """
    Sends (Test Case ID, Test Case Description, Test Case Type, Actual_Response) for
    every test case in ONE batched LLM call and gets back a Pass/Fail verdict + short
    remark for each. Returns a dict keyed by Test Case ID for easy lookup by the
    caller: { "TC_API_001": {"Pass_Fail": "Pass", "Remarks": "..."} , ... }
    """
    slim_input = [
        {
            "Test Case ID": tc.get("Test Case ID", ""),
            "Test Case Description": tc.get("Test Case Description", ""),
            "Test Case Type": tc.get("Test Case Type", ""),
            "Actual_Response": str(tc.get("Actual_Response", "")),
        }
        for tc in testcases
    ]

    user_message = (
        "Evaluate the following executed API test cases and return a Pass/Fail "
        "verdict for each, following the system prompt rules strictly.\n\n"
        f"{json.dumps(slim_input, indent=2, ensure_ascii=False)}\n\n"
        "Return the JSON array only, same order, one entry per test case."
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Evaluating pass/fail (attempt {attempt}/{max_retries})…")
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": EVALUATE_TESTCASE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=16000,
            )
            content = _strip_json_fences(response.choices[0].message.content)
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("LLM response is not a JSON array")

            result: Dict[str, Dict[str, str]] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                tc_id = str(item.get("Test Case ID", "")).strip()
                if not tc_id:
                    continue
                verdict = str(item.get("Pass_Fail", "")).strip().capitalize()
                if verdict not in ("Pass", "Fail"):
                    verdict = "Fail"
                result[tc_id] = {
                    "Pass_Fail": verdict,
                    "Remarks": str(item.get("Remarks", "")).strip(),
                }

            if not result:
                raise ValueError("No valid verdicts parsed from LLM response")

            print(f"    ✓ Evaluated {len(result)} test cases")
            return result

        except Exception as e:
            last_error = e
            print(f"    ✗ Attempt {attempt} failed: {e}")

    raise RuntimeError(f"Pass/Fail evaluation failed after {max_retries} attempts: {last_error}")

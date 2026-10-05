"""
api/endpoints/v1/agent_api.py

Endpoints used by the QA Pariksha Desktop Agent (Electron).

The agent sends the actual HTTP request to the EIS gateway FROM THE TESTER'S
DESKTOP, but the EIS keys never leave this server ("hybrid" model):

    agent ──POST /agent/eis/prepare-batch──▶ server  (encrypt + sign + RRN)
    agent ──POST {envelope} ───────────────▶ EIS gateway
    agent ──POST /agent/eis/decode-batch───▶ server  (decrypt + verify)

Generate and Pass/Fail evaluation keep using the existing endpoints
(/generate-api-testcases and /evaluate-api-testcases) unchanged.

Endpoints (all under /api/v1/testcase-generation):
    GET  /agent/client-config      — public: allowed EIS hosts, limits, min app version
    POST /agent/auth/refresh       — new JWT for the current user (old one revoked)
    POST /agent/eis/prepare-batch  — encrypted + signed envelopes, one per test case
    POST /agent/eis/decode-batch   — decrypted "Actual_Response" text, one per response

.env (optional — defaults shown):
    EIS_ALLOWED_HOSTS=eissiuat.sbi.co.in     # comma-separated; prepare refuses any other host
    AGENT_MAX_BATCH=500
    AGENT_MIN_VERSION=1.0.0
"""
import asyncio
import json
import logging
import os
import uuid as uuid_lib
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Optional
from urllib.parse import urlparse

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.config.config import settings as SETTINGS
from api.config.database import SessionDep
from api.config.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES, ALLOWED_ALGORITHMS, SECRET_KEY,
    _extract_token, blacklist_token, create_access_token, get_current_active_user,
)
from api.model import User
from api.utils.api_testcase_utils import make_rrn
from api.utils.eis_api_utils import decode_eis_response, prepare_eis_request

logger = logging.getLogger(__name__)

EIS_ALLOWED_HOSTS = {
    h.strip().lower()
    for h in os.getenv("EIS_ALLOWED_HOSTS", "eissiuat.sbi.co.in").split(",")
    if h.strip()
}
AGENT_MAX_BATCH   = int(os.getenv("AGENT_MAX_BATCH", "500"))
AGENT_MIN_VERSION = os.getenv("AGENT_MIN_VERSION", "1.0.0")

agent_router = APIRouter(
    prefix=f"/api/{SETTINGS.API_VERSION}/{SETTINGS.API_URL_PREFIX}/agent",
    tags=["Desktop Agent"],
)


# ── Request models ────────────────────────────────────────────────────────────
class PrepareBatchRequest(BaseModel):
    testcases: List[Dict[str, Any]]
    api_url: str = ""                                   # fallback when a row has no API_URL
    generated_reference_number: Optional[str] = None    # fallback RRN (same as /run-api-testcases)


class RawEisResponse(BaseModel):
    test_case_id: str = Field(..., alias="Test Case ID")
    raw: Any = None                                     # JSON body exactly as EIS returned it

    model_config = {"populate_by_name": True}


class DecodeBatchRequest(BaseModel):
    responses: List[RawEisResponse]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _host_allowed(url: str) -> Optional[str]:
    """Returns an error string if the URL may not be signed for, else None."""
    try:
        p = urlparse(url)
    except Exception:
        return f"ERROR: Invalid API URL '{url}'."
    if p.scheme not in ("https", "http") or not p.hostname:
        return f"ERROR: Invalid API URL '{url}'."
    if p.hostname.lower() not in EIS_ALLOWED_HOSTS:
        return (f"ERROR: Host '{p.hostname}' is not an allowed EIS host for the desktop agent "
                f"(allowed: {', '.join(sorted(EIS_ALLOWED_HOSTS))}).")
    return None


def _prepare_one(tc: Dict[str, Any], api_url: str, generated_reference_number: Optional[str]) -> Dict[str, Any]:
    """Same payload parsing + RRN resolution as _run_one_api_testcase() in
    generate_tests_api.py, but returns the envelope instead of sending it."""
    tc_id = tc.get("Test Case ID", "")

    raw_payload = tc.get("Test Data") or {}
    if isinstance(raw_payload, str):
        try:
            raw_payload = json.loads(raw_payload)
        except Exception:
            raw_payload = {}
    payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}

    url = tc.get("API_URL") or api_url
    base = {"Test Case ID": tc_id, "url": url, "headers": None, "body": None, "error": None}

    host_error = _host_allowed(url)
    if host_error:
        return {**base, "error": host_error}

    user_supplied_rrn = payload.get("REQUEST_REFERENCE_NUMBER")
    if user_supplied_rrn:
        rrn = user_supplied_rrn
    else:
        source_id = payload.get("SOURCE_ID")
        if source_id:
            try:
                rrn = make_rrn(str(source_id))
            except ValueError as e:
                return {**base, "error": f"ERROR: {e}"}
        else:
            rrn = generated_reference_number or ""

    prepared = prepare_eis_request(json.dumps(payload, ensure_ascii=False), rrn)
    if prepared["error"]:
        return {**base, "error": prepared["error"]}
    return {**base, "headers": prepared["headers"], "body": prepared["body"]}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@agent_router.get("/client-config")
async def client_config():
    """Public — lets the agent check compatibility before login."""
    return {
        "min_agent_version": AGENT_MIN_VERSION,
        "eis_allowed_hosts": sorted(EIS_ALLOWED_HOSTS),
        "max_batch": AGENT_MAX_BATCH,
        "token_expire_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
    }


@agent_router.post("/auth/refresh")
async def refresh_token(
    request: Request,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Issues a fresh JWT for the current user and revokes the old one, so a long
    generate → run → review → evaluate session doesn't hit a 401 mid-way."""
    from api.endpoints.v1.user_management_api import _user_jwt_payload   # avoid import cycle at load

    old = await _extract_token(request)
    try:
        p = jwt.decode(old, SECRET_KEY, algorithms=ALLOWED_ALGORITHMS, options={"verify_exp": False})
        if p.get("jti"):
            exp = p.get("exp")
            blacklist_token(
                p["jti"], current_user.username,
                datetime.fromtimestamp(exp, tz=timezone.utc) if exp
                else datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
                session,
            )
    except Exception as e:
        logger.warning(f"Agent refresh: could not revoke old token — {e}")

    new_token = create_access_token(
        data=_user_jwt_payload(current_user, str(uuid_lib.uuid4())),
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": new_token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}


@agent_router.post("/eis/prepare-batch")
async def prepare_batch(
    request: PrepareBatchRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Encrypt + sign every test case. The agent POSTs each `body` with `headers`
    to `url` itself. Rows that can't be prepared carry an `error` string, which the
    agent shows as that row's Actual_Response (same text the server-side run gives)."""
    if not request.testcases:
        raise HTTPException(422, "No test cases provided.")
    if len(request.testcases) > AGENT_MAX_BATCH:
        raise HTTPException(413, f"Too many test cases in one batch (max {AGENT_MAX_BATCH}).")

    prepared = await asyncio.to_thread(
        lambda: [_prepare_one(tc, request.api_url, request.generated_reference_number) for tc in request.testcases]
    )
    logger.info(f"Agent prepare-batch: user={current_user.username} rows={len(prepared)} "
                f"errors={sum(1 for p in prepared if p['error'])}")
    return {"prepared": prepared}


@agent_router.post("/eis/decode-batch")
async def decode_batch(
    request: DecodeBatchRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Decrypt + verify what EIS returned. Only send rows where EIS actually
    returned a JSON body — network errors / timeouts are reported by the agent itself."""
    if len(request.responses) > AGENT_MAX_BATCH:
        raise HTTPException(413, f"Too many responses in one batch (max {AGENT_MAX_BATCH}).")

    decoded = await asyncio.to_thread(
        lambda: [{"Test Case ID": r.test_case_id, "actual_response": decode_eis_response(r.raw)}
                 for r in request.responses]
    )
    return {"decoded": decoded}

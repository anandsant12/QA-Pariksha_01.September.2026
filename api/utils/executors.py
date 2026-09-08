"""
api/utils/executors.py

Bounded, purpose-specific thread pools used to run blocking/synchronous work
(LLM calls, RAG/ChromaDB retrieval, password hashing, outbound HTTPS to the
EIS API) OFF the single asyncio event loop — without letting a burst of
concurrent requests spawn unbounded threads and push the box's CPU/memory to
its limit.

WHY THIS EXISTS
----------------
main.py pins the app to a single Uvicorn worker (one process, one event
loop) — required because the ChromaDB PersistentClient cannot safely be
forked across multiple worker processes. That means ALL concurrency across
every user has to happen via threads within this one process. If we just
declared endpoints "async def" and called blocking code directly (the
original state of this codebase), a single slow call — an LLM request, a
document being OCR'd, a loop of outbound EIS calls — froze the event loop
entirely, so NO other user's request (not even a login) could be served
until it finished.

Routing blocking work through `loop.run_in_executor(POOL, func, *args)`
fixes the freeze — but an *unbounded* pool (or one pool shared by everything)
just trades "the app freezes" for "50 people at once spin up 50 threads all
doing memory-hungry work simultaneously and the box falls over instead."
So each pool below is sized deliberately, based on the resource profile of
the work it carries, and requests beyond that size simply queue (fast, cheap,
in-memory) rather than spawning more threads or blocking the event loop.

POOLS
-----
  LLM_WORK_POOL — CPU/memory-heavy: LLM calls, ChromaDB/RAG retrieval,
                  document/page processing, dedup. Kept SMALL. Each of these
                  can hold a meaningful amount of data in memory at once
                  (page text, embeddings, LLM responses) — we deliberately
                  cap how many run at the same time regardless of how many
                  users are active, rather than letting concurrent document
                  uploads or test-generation requests pile up memory use.

  EIS_CALL_POOL — I/O-bound: outbound HTTPS calls to the bank's EIS API.
                  CPU/memory cost per call is small (AES/RSA on a short
                  payload, then waiting on the network) — but MORE workers
                  here also means MORE simultaneous load on the downstream
                  EIS UAT/SIT environment, which is outside our control. The
                  size below is a deliberate rate-limit as much as a
                  resource-limit: it bounds how many test cases (across ALL
                  users, not just one request) can be in flight against the
                  real bank API at once.

  AUTH_POOL     — CPU-bound but fast: password hashing/verification
                  (argon2, intentionally slow/memory-hard by design so it
                  resists brute-forcing). Capped so a burst of logins at the
                  start of the working day can't spike CPU by all hashing at
                  once.

TUNING
------
Every size below is overridable via an environment variable, so ops can
adjust to the actual box's CPU count / RAM without a code change — e.g. a
bigger VM can raise LLM_WORK_POOL_SIZE; a smaller one should probably lower
it further. Restart the app for a change to take effect (pools are created
once, at import time).
"""
import os
from concurrent.futures import ThreadPoolExecutor


def _pool_size(env_var: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(env_var, str(default))))
    except ValueError:
        return default


LLM_WORK_POOL_SIZE = _pool_size("LLM_WORK_POOL_SIZE", 4)
EIS_CALL_POOL_SIZE  = _pool_size("EIS_CALL_POOL_SIZE", 8)
AUTH_POOL_SIZE      = _pool_size("AUTH_POOL_SIZE", 8)

LLM_WORK_POOL = ThreadPoolExecutor(max_workers=LLM_WORK_POOL_SIZE, thread_name_prefix="llm-work")
EIS_CALL_POOL = ThreadPoolExecutor(max_workers=EIS_CALL_POOL_SIZE, thread_name_prefix="eis-call")
AUTH_POOL     = ThreadPoolExecutor(max_workers=AUTH_POOL_SIZE, thread_name_prefix="auth")

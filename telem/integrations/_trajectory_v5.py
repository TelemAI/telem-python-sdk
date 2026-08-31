"""Framework-neutral identity helpers for the trajectory-v5 wire protocol.

Also home to the framework-neutral half of **incremental history transmission,
phase 1**: the delivered/capability caches, the send-time omission decision, and the
guard's 409 discriminator. It stays identity-shaped — no client, no harness, no
HTTP — so each surface keeps its own state and its own call sites and shares the
rules rather than re-implementing them.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid5

NS_TRAJECTORY = UUID("443866ab-1b45-5ed8-979e-52fdad07b810")
NONE = "none"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _lp(value: str) -> str:
    return f"{len(value.encode())}:{value}"


def fingerprint(harness: str, conversation_id: str) -> str:
    """Return a stable, non-reversible conversation identity."""
    return _sha256(_lp(harness) + _lp(conversation_id))


def session_key(harness: str, conversation_id: str, window_id: str) -> str:
    """Return the deterministic identity of one context-window generation."""
    name = _lp(harness) + _lp(_sha256(conversation_id)) + _lp(window_id) + _lp(NONE)
    return str(uuid5(NS_TRAJECTORY, name))


def event_node_key(
    harness: str,
    session: str,
    message_id: str,
    tool_call_id: str,
) -> str:
    """Return a replay-stable event node key for one model tool call."""
    name = _lp(harness) + _lp(session) + _lp(message_id) + _lp(tool_call_id) + _lp("event")
    return str(uuid5(NS_TRAJECTORY, name))


def snapshot_node_key(harness: str, conversation_id: str, message_id: str) -> str:
    """Return the identity of an agent snapshot at a delegation boundary."""
    name = _lp(harness) + _lp(_sha256(conversation_id)) + _lp(message_id) + _lp("snap")
    return str(uuid5(NS_TRAJECTORY, name))


HISTORY_TEXT_CAP = 128_000
TOOL_INPUT_CAP = 128_000

#: Metadata keys the trajectory payload owns. A caller-supplied value for any of
#: these is dropped rather than allowed to forge identity.
RESERVED_METADATA = frozenset(
    {
        "message_history",
        "session_key",
        "fingerprint",
        "node_key",
        "kind",
        "parent_node_key",
        "ancestors",
    }
)


def content_anchor(*parts: str) -> str:
    """Return a stable anchor id for content a host did not assign an id to.

    Length-prefixed like every other key component, so concatenation stays injective.
    """
    return _sha256("".join(_lp(part) for part in parts))


def tool_marker(name: str, status: str, args: Any) -> str:
    """Render one tool call as the compact acting-trace marker used in history."""
    try:
        serialized = json.dumps(args, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        serialized = ""
    suffix = f" {serialized[:TOOL_INPUT_CAP]}" if serialized else ""
    return f"[tool {name or '?'}: {status}{suffix}]"


def build_metadata(
    *,
    harness: str,
    conversation_id: str,
    window_id: str,
    message_id: str,
    tool_call_id: str,
    history: list[dict[str, str]],
    ancestors: list[dict[str, Any]],
    kind: str = "search",
    caller_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the trajectory-v5 metadata block for one event node.

    Caller metadata is preserved except for the reserved keys, which the generated
    values always win. ``parent_node_key`` is the DIRECT parent's snapshot key — the
    last entry of the root-first ancestor chain — or ``None`` when there is no parent.
    """
    metadata = {
        key: value for key, value in (caller_metadata or {}).items() if key not in RESERVED_METADATA
    }
    session = session_key(harness, conversation_id, window_id)
    metadata.update(
        {
            "message_history": history,
            "session_key": session,
            "fingerprint": fingerprint(harness, conversation_id),
            "node_key": event_node_key(harness, session, message_id, tool_call_id),
            "kind": kind,
            "parent_node_key": ancestors[-1].get("node_key") if ancestors else None,
            "ancestors": ancestors,
        }
    )
    return metadata


def build_snapshot(
    *,
    harness: str,
    conversation_id: str,
    window_id: str,
    message_id: str,
    context: list[dict[str, str]],
    ancestors: list[dict[str, Any]],
    spawned_at: str | None = None,
    context_omitted: bool = False,
) -> dict[str, Any]:
    """Freeze one agent at a delegation boundary as an ancestor entry.

    With ``context_omitted=True`` the entry carries ``context_omitted: True`` and
    **no** ``context`` key, in the same position — every other key, ``spawned_at``
    included, is unchanged (the contract: the entry is unchanged on every call except
    the ``context`` key). ``context`` is still a required argument, because the
    caller has to have materialized it to be able to restore it on the guard's 409;
    withholding is a send-time decision, never a "did not build it" one.

    WHICH snapshots may be omitted is the caller's decision, not this builder's —
    it takes proof of delivery under the current scope, which lives on the surface
    (see :func:`plan_ancestors`).
    """
    return {
        "session_key": session_key(harness, conversation_id, window_id),
        "fingerprint": fingerprint(harness, conversation_id),
        "node_key": snapshot_node_key(harness, conversation_id, message_id),
        "parent_node_key": ancestors[-1].get("node_key") if ancestors else None,
        **({CONTEXT_OMITTED: True} if context_omitted else {"context": context}),
        "spawned_at": spawned_at,
    }


# --------------------------------------------------------------------------- #
# Incremental history transmission, phase 1 — an ancestor's context travels ONCE
# per snapshot. Today every call re-sends the same flattened ancestor history
# byte for byte, and the backend is first-writer-wins on node content, so every
# re-send after the first is already a no-op.
#
# What makes dropping it safe is PROOF, not byte counting. An ancestor node whose
# FIRST arrival carries no context is stored with a null context frozen forever —
# unrepairable in the Telem backend and in obs, and it empties the conversation title of a
# purely-delegating root. So context is omitted only for a snapshot this surface
# has PROVEN delivered and only to a backend that has PROVEN it
# implements the contract guard. Both proofs are per ``(base_url, key)``
# scope, because node ids are derived from the account key server-side: flip
# either and every belief about that world is void.
# --------------------------------------------------------------------------- #

#: The marker that turns an absent ``context`` into a licence rather than a null.
CONTEXT_OMITTED = "context_omitted"
#: The response key that is the capability probe, and the guard's typed 409 code.
MISSING_SNAPSHOTS = "missing_snapshots"

#: Snapshot keys are one-way hashes that can never be mapped back to a session,
#: so idle-eviction is impossible and the cache is bounded by count alone
#: by count. An eviction costs one redundant full re-send — the safe direction.
DELIVERED_CAP = 4096
#: A process sees one or two scopes in its life. Bounded anyway; forgetting a
#: capability just re-learns it from the next response, and until then nothing is
#: omitted — again the safe direction.
CAPABILITY_CAP = 64

#: ``TELEM_INCREMENTAL`` values. Phase 1 is ON by default (owner decision
#: 2026-08-26): unset, unrecognized, and ``history`` all resolve to
#: :data:`MODE_ANCESTORS`. ``history`` is phase 2, which is opencode-only — on the
#: Python surfaces it buys phase 1 and nothing more, rather than erroring.
MODE_OFF = "off"
MODE_ANCESTORS = "ancestors"


class BoundedKeySet:
    """A bounded, insertion-ordered, lock-guarded set of opaque keys.

    Re-adding refreshes recency; the oldest key is the first evicted. The lock is
    the point: the openai wrap's state is shared by whatever threads drive one
    client, and the Hermes plugin's is process-global with hooks firing on threads
    the tool handler never runs on.
    """

    __slots__ = ("_lock", "_keys", "_cap")

    def __init__(self, cap: int) -> None:
        self._lock = threading.Lock()
        self._keys: OrderedDict[str, bool] = OrderedDict()
        self._cap = max(1, cap)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._keys

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)

    def add(self, key: str) -> None:
        with self._lock:
            self._keys.pop(key, None)
            self._keys[key] = True
            while len(self._keys) > self._cap:
                self._keys.popitem(last=False)

    def discard(self, key: str) -> None:
        with self._lock:
            self._keys.pop(key, None)


def cache_scope(base_url: str | None, api_key: str | None) -> str:
    """The world a delivery belief holds in: base URL plus a HASHED key.

    The key is hashed here and nowhere held or logged raw — this string ends up as
    a cache key, never on the wire and never in a warning.
    """
    return f"{base_url or ''} {_sha256(api_key or '')}"


def delivered_key(scope: str, snapshot_key: str) -> str:
    """One flat cache key over (scope, snapshot), so the bound is a true total.

    A scope flip then simply misses instead of needing its own eviction policy.
    NUL is not producible by either component (a hashed url+key and a uuid).
    """
    return f"{scope}\x00{snapshot_key}"


def incremental_mode(env: Mapping[str, str] | None = None) -> str:
    """Resolve ``TELEM_INCREMENTAL`` for THIS call.

    Read per call, like the rest of the config: flipping it takes effect on the
    next search, with no restart. An unrecognized value resolves to the DEFAULT,
    not to ``off`` — the rollback lever is the exact word ``off`` (after trim and
    lowercase) and nothing else, so an operator reaching for it should verify the
    value that landed rather than the intent.
    """
    source = os.environ if env is None else env
    raw = (source.get("TELEM_INCREMENTAL") or "").strip().lower()
    return MODE_OFF if raw == MODE_OFF else MODE_ANCESTORS


def force_capability(env: Mapping[str, str] | None = None) -> bool:
    """``TELEM_INCREMENTAL_FORCE=1`` — the test-only bypass of the capability probe.

    Not a second mode: it skips the server's own signal, which only a
    harness whose backend is correct by construction may do. Never production.
    """
    source = os.environ if env is None else env
    return source.get("TELEM_INCREMENTAL_FORCE") == "1"


def omits_delivered_context(
    scope: str,
    capability: BoundedKeySet,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Does THIS call withhold context for snapshots already delivered to *scope*?

    On by default is safe because the mode is only HALF the gate: omitting also
    requires this exact scope to have answered with the ``missing_snapshots`` KEY,
    which only a backend carrying the contract guard emits. Against an old or
    third-party backend the probe never fires, the surface keeps sending full
    contexts — byte-identical to the pre-wave wire — and the default can never
    strand a node whose context nobody stored. ``off`` short-circuits the mode
    half (before the force bypass) and stays the instant, no-deploy rollback lever.
    """
    if incremental_mode(env) == MODE_OFF:
        return False
    if force_capability(env):
        return True
    return scope in capability


@dataclass
class Withheld:
    """One ancestor entry this call chose not to send the context for.

    ``entry`` is the very dict riding in ``metadata["ancestors"]``, so restoring is
    an in-place swap on the body that is about to be re-serialized — same
    ``node_key``, same everything else.
    """

    node_key: str
    entry: dict[str, Any]
    context: Any


@dataclass
class DeliveryPlan:
    """The two unacknowledged promises one call makes, until the response decides.

    ``sent_with_context`` is exactly what a 2xx proves delivered; ``omitted`` is
    what the guard's 409 asks back. Empty ``omitted`` is also the gate: a 409 on a
    request that withheld nothing is somebody else's error, never a retry.
    """

    scope: str
    sent_with_context: list[str] = field(default_factory=list)
    omitted: list[Withheld] = field(default_factory=list)


def omit_context(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return *entry* with ``context`` replaced, in place, by ``context_omitted``."""
    withheld: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "context":
            withheld[CONTEXT_OMITTED] = True
        else:
            withheld[key] = value
    return withheld


def plan_ancestors(
    ancestors: Iterable[Mapping[str, Any]],
    plan: DeliveryPlan,
    *,
    delivered: BoundedKeySet,
    capability: BoundedKeySet,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return the ancestor list to SEND, recording this call's promises on *plan*.

    Every entry is copied first, so the surface's own stored chain is never
    mutated by a send decision. An entry with no ``context`` key (nothing to
    withhold) or no ``node_key`` (no identity to prove a delivery against) is
    passed through and promises nothing.
    """
    omitting = omits_delivered_context(plan.scope, capability, env)
    planned: list[dict[str, Any]] = []
    for original in ancestors:
        entry = dict(original)
        node_key = str(entry.get("node_key") or "")
        if "context" not in entry or not node_key:
            planned.append(entry)
            continue
        if omitting and delivered_key(plan.scope, node_key) in delivered:
            context = entry["context"]
            entry = omit_context(entry)
            plan.omitted.append(Withheld(node_key=node_key, entry=entry, context=context))
        else:
            plan.sent_with_context.append(node_key)
        planned.append(entry)
    return planned


def record_delivery(
    plan: DeliveryPlan,
    body: Any,
    *,
    delivered: BoundedKeySet,
    capability: BoundedKeySet,
) -> None:
    """Fold a proven 2xx into the caches. Never call this at send time.

    Send-time marking lets a parallel sibling omit while the first request can
    still fail before the backend's Phase B commits, which is exactly how a
    permanently context-less node is born.

    Marking runs even when the mode is ``off``: passive learning costs nothing, is
    never acted on while off, and means flipping the mode on needs no warm-up call
    to re-earn what this surface already proved.
    """
    for node_key in plan.sent_with_context:
        delivered.add(delivered_key(plan.scope, node_key))
    if not isinstance(body, Mapping):
        return
    # KEY PRESENCE, never truthiness — the healthy value is `[]`. A
    # backend without the guard omits the key entirely and so never grants
    # capability, which is what keeps an omitting client off an old server.
    if MISSING_SNAPSHOTS not in body:
        return
    capability.add(plan.scope)
    # Reported keys are nodes the backend refused to create: un-mark them so the
    # next call carries their full context again. Under form 1c the list
    # is always empty, but the field is contractual and this is one line.
    missing = body[MISSING_SNAPSHOTS]
    if isinstance(missing, list):
        for node_key in missing:
            delivered.discard(delivered_key(plan.scope, str(node_key)))


def restore_omitted(plan: DeliveryPlan, *, delivered: BoundedKeySet) -> None:
    """Put every withheld context back into the body this call already built.

    Un-marking comes FIRST and is not conditional on the retry succeeding: the 409
    falsified this surface's belief that those rows exist, and a sibling building
    its body right now must carry their contexts too. A retry that then fails
    leaves them unmarked, which is the safe direction — one redundant full re-send.

    The keys move to ``sent_with_context``, so the retry's 2xx marks exactly what
    the retry actually carried, through the same :func:`record_delivery` as any
    call. ``omitted`` is then empty, which is what makes a second 409 unable to
    re-enter the retry branch even in principle.
    """
    for withheld in plan.omitted:
        delivered.discard(delivered_key(plan.scope, withheld.node_key))
        restored: dict[str, Any] = {}
        for key, value in withheld.entry.items():
            if key == CONTEXT_OMITTED:
                restored["context"] = withheld.context
            else:
                restored[key] = value
        # In place, and position-preserving: the entry object is the one inside
        # the metadata the caller is about to re-send.
        withheld.entry.clear()
        withheld.entry.update(restored)
        plan.sent_with_context.append(withheld.node_key)
    plan.omitted = []


def refusal_code(body: Any) -> str | None:
    """The typed error code, from either envelope the two doors use.

    FastAPI's ``detail`` on ``/v1/interactions``, the ``{"error": {...}}`` envelope
    on ``/v1/fetch``. The bare top-level read is pure defense — a proxy that
    unwraps, a future door that does not wrap. ``None`` means "this body names no
    code we can read", which includes a body that is not JSON at all: the SDK hands
    over ``response.text`` when the body would not parse.
    """
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    if not isinstance(body, Mapping):
        return None
    for shape in (body.get("detail"), body.get("error"), body):
        if isinstance(shape, Mapping):
            code = shape.get("code")
            if isinstance(code, str) and code:
                return code
    return None


def is_missing_snapshots_refusal(status_code: int | None, body: Any, plan: DeliveryPlan) -> bool:
    """Is this 409 the guard asking for the withheld contexts back?

    Two judgement calls, both deliberate, because a defensive parse has to decide
    what an ambiguous body means:

    * **No code readable** (unparseable body, a proxy's plain-text 409, an envelope
      shape we do not know) — TREAT IT AS THE REFUSAL. The retry is a standard full
      request re-sending the same ``node_key``s, so a wrong guess costs one round
      trip; guessing the other way costs the user a failed tool call on the one
      condition this whole path exists to heal.
    * **A code that is present and is NOT** ``missing_snapshots`` — not the refusal.
      The server named a different reason, restoring contexts cannot address it,
      and re-POSTing a request it just refused for a stated reason is a blind retry
      of a non-idempotent call. Surface it.

    And the gate above both: this request must have omitted something.
    """
    if status_code != 409 or not plan.omitted:
        return False
    code = refusal_code(body)
    return code is None or code == MISSING_SNAPSHOTS

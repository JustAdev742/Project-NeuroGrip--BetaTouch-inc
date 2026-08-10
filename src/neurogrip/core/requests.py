"""Request/response over the event bus.

:mod:`neurogrip.core.events` gives fire-and-forget publish/subscribe, which is
the right default: it keeps producers ignorant of consumers and never blocks the
control loop. But some interactions genuinely need an answer — "what is the
current calibration?", "run the servo range test and tell me the result",
"is the depth camera present?" — and expressing those as two one-way topics
leaves every caller hand-rolling correlation and timeout logic.

This adds a thin request/response layer *on top of* the existing bus rather than
beside it, so:

* responders are still discovered by topic, not by import — a requester never
  holds a reference to the service that answers;
* every request and reply still appears in bus history, so the black-box
  recorder and diagnostics console see the whole conversation;
* there is exactly one bus to reason about.

Two rules keep this from becoming a distributed-systems footgun in a device that
must not stall:

**Every request has a timeout, and it is mandatory.** A request with no deadline
is a latent hang. The default is deliberately short.

**Responders must never be called from the control loop.** The bus delivers
synchronously, so a slow responder blocks the requester's thread. Anything that
touches hardware or disk registers through :class:`QueuedSubscriber` and replies
asynchronously — the correlation machinery here handles the late reply correctly.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .clock import Clock, RealClock
from .errors import NeuroGripError
from .events import Event, EventBus, Subscription
from .logging import get_logger

__all__ = [
    "NoResponder",
    "Request",
    "RequestBroker",
    "RequestError",
    "RequestTimeout",
    "Response",
]

log = get_logger(__name__)

#: Suffix appended to a request topic to form its reply topic.
_REPLY_SUFFIX = ".reply"

#: Default deadline. Chosen to be shorter than the shortest watchdog in the
#: system, so a wedged responder trips a request timeout before it trips a
#: safety fault — the former is diagnosable, the latter just stops the hand.
DEFAULT_TIMEOUT_S = 2.0


class RequestError(NeuroGripError):
    """Base for request failures."""


class RequestTimeout(RequestError):
    """No reply arrived before the deadline."""


class NoResponder(RequestError):
    """Nothing is registered to answer this topic.

    Distinguished from a timeout on purpose: "nobody is listening" is a wiring
    bug you fix at build time, while "listener was too slow" is a runtime
    condition. Collapsing them into one error makes the first one very hard to
    find.
    """


@dataclass(frozen=True, slots=True)
class Request:
    """An outbound request. Delivered as the payload of the request topic."""

    topic: str
    payload: Any
    correlation_id: str
    reply_topic: str
    deadline: float
    source: str = ""


@dataclass(frozen=True, slots=True)
class Response:
    """A reply to a :class:`Request`."""

    correlation_id: str
    payload: Any = None
    error: str = ""
    source: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass(slots=True)
class _Pending:
    """Server-side bookkeeping for one in-flight request."""

    event: threading.Event = field(default_factory=threading.Event)
    response: Response | None = None


class RequestBroker:
    """Correlates requests with replies on an :class:`EventBus`.

    Example::

        broker = RequestBroker(bus, clock)

        # service side
        broker.respond("servo.limits", lambda req: bus_limits)

        # caller side, anywhere, with no import of the servo package
        limits = broker.request("servo.limits", timeout=0.5)
    """

    def __init__(
        self,
        bus: EventBus,
        clock: Clock | None = None,
        *,
        source: str = "",
    ) -> None:
        self._bus = bus
        self._clock: Clock = clock or RealClock()
        self._source = source
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._responders: dict[str, Subscription] = {}
        self._reply_subs: dict[str, Subscription] = {}
        self.requests_sent = 0
        self.requests_served = 0
        self.timeouts = 0

    # -- responder side -------------------------------------------------------

    def respond(
        self,
        topic: str,
        handler: Callable[[Any], Any],
        *,
        name: str = "",
    ) -> Subscription:
        """Register ``handler`` as the responder for ``topic``.

        The handler receives the request *payload* (not the envelope) and returns
        the reply payload. Raising is fine and expected — the exception is
        converted into an error response, so a responder bug surfaces at the
        caller instead of vanishing into the bus's handler-error counter.
        """
        if topic in self._responders:
            raise RequestError(f"a responder for '{topic}' is already registered")

        def _on_request(event: Event) -> None:
            request = event.payload
            if not isinstance(request, Request):
                return
            try:
                result = handler(request.payload)
                response = Response(
                    correlation_id=request.correlation_id,
                    payload=result,
                    source=name or topic,
                )
            except Exception as exc:
                log.warning(
                    "request responder failed",
                    topic=topic,
                    error=f"{type(exc).__name__}: {exc}",
                )
                response = Response(
                    correlation_id=request.correlation_id,
                    error=f"{type(exc).__name__}: {exc}",
                    source=name or topic,
                )
            self.requests_served += 1
            self._bus.publish(request.reply_topic, response, source=name or topic)

        sub = self._bus.subscribe(topic, _on_request, name=f"responder:{topic}")
        self._responders[topic] = sub
        return sub

    def unregister(self, topic: str) -> None:
        """Remove the responder for ``topic``. Idempotent."""
        sub = self._responders.pop(topic, None)
        if sub is not None:
            sub.cancel()

    def has_responder(self, topic: str) -> bool:
        return topic in self._responders

    # -- caller side ----------------------------------------------------------

    def request(
        self,
        topic: str,
        payload: Any = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        require_responder: bool = True,
    ) -> Any:
        """Send a request and wait for the reply.

        Raises :class:`NoResponder` if nothing is registered, :class:`RequestTimeout`
        if the deadline passes, and :class:`RequestError` if the responder raised.
        """
        if require_responder and not self.has_responder(topic):
            raise NoResponder(f"no responder registered for '{topic}'")
        if timeout <= 0:
            raise RequestError("timeout must be positive; a request without a deadline can hang")

        correlation_id = uuid.uuid4().hex
        reply_topic = f"{topic}{_REPLY_SUFFIX}"
        pending = _Pending()

        with self._lock:
            self._pending[correlation_id] = pending
            # One reply subscription per topic, shared by all in-flight requests
            # on it — subscribing per request would churn the bus's subscription
            # lists at request rate.
            if reply_topic not in self._reply_subs:
                self._reply_subs[reply_topic] = self._bus.subscribe(
                    reply_topic, self._on_reply, name=f"reply:{topic}"
                )

        request = Request(
            topic=topic,
            payload=payload,
            correlation_id=correlation_id,
            reply_topic=reply_topic,
            deadline=self._clock.monotonic() + timeout,
            source=self._source,
        )
        self.requests_sent += 1

        try:
            # Synchronous bus delivery means an in-thread responder has already
            # replied by the time publish() returns, and the wait below falls
            # straight through. The Event is for responders that reply from
            # another thread.
            self._bus.publish(topic, request, source=self._source)
            if not pending.event.wait(timeout):
                self.timeouts += 1
                raise RequestTimeout(
                    f"no reply on '{topic}' within {timeout:.2f}s",
                    context={"topic": topic, "correlation_id": correlation_id},
                )
        finally:
            with self._lock:
                self._pending.pop(correlation_id, None)

        response = pending.response
        if response is None:  # pragma: no cover - set before the event fires
            raise RequestTimeout(f"empty reply on '{topic}'")
        if not response.ok:
            raise RequestError(response.error, context={"topic": topic})
        return response.payload

    def try_request(
        self,
        topic: str,
        payload: Any = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        default: Any = None,
    ) -> Any:
        """Like :meth:`request` but returns ``default`` instead of raising.

        For callers on the UI or telemetry path, where a missing answer should
        degrade the display rather than propagate an exception into a render.
        """
        try:
            return self.request(topic, payload, timeout=timeout)
        except RequestError:
            return default

    def _on_reply(self, event: Event) -> None:
        response = event.payload
        if not isinstance(response, Response):
            return
        with self._lock:
            pending = self._pending.get(response.correlation_id)
        if pending is None:
            # A reply that arrived after its deadline. Dropping it is correct —
            # the caller has already given up — but it is worth counting, since
            # a steady stream means a timeout is set too tight.
            return
        pending.response = response
        pending.event.set()

    # -- introspection --------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            in_flight = len(self._pending)
        return {
            "sent": self.requests_sent,
            "served": self.requests_served,
            "timeouts": self.timeouts,
            "in_flight": in_flight,
            "responders": sorted(self._responders),
        }

    def close(self) -> None:
        """Cancel every subscription. Idempotent."""
        for sub in list(self._responders.values()):
            sub.cancel()
        self._responders.clear()
        for sub in list(self._reply_subs.values()):
            sub.cancel()
        self._reply_subs.clear()
        with self._lock:
            for pending in self._pending.values():
                pending.event.set()
            self._pending.clear()

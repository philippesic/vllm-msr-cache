from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

MAX_NAME_LENGTH = 192
MAX_GENERATION = 2**63 - 1
MAX_TIMESTAMP = 2**63 - 1
MAX_BOUNDARY = 1_000_000
MAX_HINT_WINDOW_NS = 60_000_000_000
MAX_KEY_BYTES = 256


def bounded_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= MAX_NAME_LENGTH:
        raise ValueError(name)
    return value


def generation_number(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_GENERATION:
        raise ValueError("generation")
    return value


@dataclass(frozen=True)
class Namespace:
    model_revision: str
    tokenizer_revision: str
    adapter_namespace: str
    layout_namespace: str

    def __post_init__(self) -> None:
        for name, value in self.mapping().items():
            bounded_name(value, name)

    def mapping(self) -> dict[str, str]:
        return {
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "adapter_namespace": self.adapter_namespace,
            "layout_namespace": self.layout_namespace,
        }

    @classmethod
    def parse(cls, raw: object) -> Namespace:
        names = (
            "model_revision",
            "tokenizer_revision",
            "adapter_namespace",
            "layout_namespace",
        )
        if not isinstance(raw, dict) or set(raw) != set(names):
            raise ValueError("namespace fields")
        if any(
            not isinstance(raw[n], str) or not 0 < len(raw[n]) <= MAX_NAME_LENGTH
            for n in names
        ):
            raise ValueError("namespace values")
        return cls(*(str(raw[n]) for n in names))


def prefix_digest(tokens: Sequence[int], namespace: Namespace) -> str:
    if any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("token IDs")
    payload = json.dumps(
        {"namespace": namespace.mapping(), "token_ids": list(tokens)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def positive_integer(raw: object, name: str, maximum: int) -> int:
    if type(raw) is not int or not 0 < raw <= maximum:
        raise ValueError(name)
    return raw


@dataclass(frozen=True)
class Hint:
    scope: str
    session: str
    incarnation: str
    generation: int
    event: str
    prefix: str
    boundary: int
    namespace: Namespace
    observed_ns: int
    published_ns: int
    deadline_ns: int
    clock_domain: str

    def __post_init__(self) -> None:
        for name in (
            "scope",
            "session",
            "incarnation",
            "event",
            "prefix",
            "clock_domain",
        ):
            bounded_name(getattr(self, name), name)
        generation_number(self.generation)
        if not isinstance(self.namespace, Namespace):
            raise ValueError("namespace")
        positive_integer(self.boundary, "boundary", MAX_BOUNDARY)
        for name in ("observed_ns", "published_ns", "deadline_ns"):
            positive_integer(getattr(self, name), name, MAX_TIMESTAMP)
        if (
            not self.observed_ns <= self.published_ns < self.deadline_ns
            or self.deadline_ns - self.published_ns > MAX_HINT_WINDOW_NS
        ):
            raise ValueError("causal clocks")

    def mapping(self) -> dict[str, object]:
        return {"schema": "wait-hint-v2", **asdict(self)}

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> Hint:
        fields = {
            "schema",
            "scope",
            "session",
            "incarnation",
            "generation",
            "event",
            "prefix",
            "boundary",
            "namespace",
            "observed_ns",
            "published_ns",
            "deadline_ns",
            "clock_domain",
        }
        if set(raw) != fields or raw["schema"] != "wait-hint-v2":
            raise ValueError("hint schema/fields")
        values = {}
        for name in (
            "scope",
            "session",
            "incarnation",
            "event",
            "prefix",
            "clock_domain",
        ):
            value = raw[name]
            if not isinstance(value, str) or not 0 < len(value) <= MAX_NAME_LENGTH:
                raise ValueError(name)
            values[name] = value
        generation = raw["generation"]
        if type(generation) is not int or not 0 <= generation <= MAX_GENERATION:
            raise ValueError("generation")
        observed = positive_integer(raw["observed_ns"], "observed_ns", MAX_TIMESTAMP)
        published = positive_integer(raw["published_ns"], "published_ns", MAX_TIMESTAMP)
        deadline = positive_integer(raw["deadline_ns"], "deadline_ns", MAX_TIMESTAMP)
        if (
            not observed <= published < deadline
            or deadline - published > MAX_HINT_WINDOW_NS
        ):
            raise ValueError("causal clocks")
        return cls(
            values["scope"],
            values["session"],
            values["incarnation"],
            generation,
            values["event"],
            values["prefix"],
            positive_integer(raw["boundary"], "boundary", MAX_BOUNDARY),
            Namespace.parse(raw["namespace"]),
            observed,
            published,
            deadline,
            values["clock_domain"],
        )


@dataclass(frozen=True)
class Registration:
    scope: str
    session: str
    incarnation: str
    generation: int

    def __post_init__(self) -> None:
        for name in ("scope", "session", "incarnation"):
            bounded_name(getattr(self, name), "registration " + name)
        generation_number(self.generation)

    @classmethod
    def parse(cls, raw: object) -> Registration:
        fields = {"schema", "scope", "session", "incarnation", "generation"}
        if (
            not isinstance(raw, dict)
            or set(raw) != fields
            or raw["schema"] != "wait-registration-v2"
        ):
            raise ValueError("registration schema")
        for name in ("scope", "session", "incarnation"):
            if (
                not isinstance(raw[name], str)
                or not 0 < len(raw[name]) <= MAX_NAME_LENGTH
            ):
                raise ValueError("registration identity")
        generation = raw["generation"]
        if type(generation) is not int or not 0 <= generation <= MAX_GENERATION:
            raise ValueError("registration generation")
        return cls(raw["scope"], raw["session"], raw["incarnation"], generation)

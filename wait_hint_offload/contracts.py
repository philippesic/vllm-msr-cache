from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Namespace:
    model_revision: str
    tokenizer_revision: str
    adapter_namespace: str
    layout_namespace: str

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
            not isinstance(raw[n], str) or not 0 < len(raw[n]) <= 192 for n in names
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
            if not isinstance(value, str) or not 0 < len(value) <= 192:
                raise ValueError(name)
            values[name] = value
        generation = raw["generation"]
        if type(generation) is not int or not 0 <= generation < 2**63:
            raise ValueError("generation")
        observed = positive_integer(raw["observed_ns"], "observed_ns", 2**63 - 1)
        published = positive_integer(raw["published_ns"], "published_ns", 2**63 - 1)
        deadline = positive_integer(raw["deadline_ns"], "deadline_ns", 2**63 - 1)
        if (
            not observed <= published < deadline
            or deadline - published > 60_000_000_000
        ):
            raise ValueError("causal clocks")
        return cls(
            values["scope"],
            values["session"],
            values["incarnation"],
            generation,
            values["event"],
            values["prefix"],
            positive_integer(raw["boundary"], "boundary", 1_000_000),
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
            if not isinstance(raw[name], str) or not 0 < len(raw[name]) <= 192:
                raise ValueError("registration identity")
        generation = raw["generation"]
        if type(generation) is not int or not 0 <= generation < 2**63:
            raise ValueError("registration generation")
        return cls(raw["scope"], raw["session"], raw["incarnation"], generation)

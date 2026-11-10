# 005: Agent guardrails

Accepted for the non-source stage. Based on the garden audit summarized in the
approved task: contradictory status writers, ad-hoc launchers, silent fallbacks,
unused generated programs, interpreter drift and unreasoned suppressions.
Historical evidence remains indexed in [004](004-evidence.md) and
[FINDINGS](../../FINDINGS.md); no runtime completion follows from this decision.

## G005-COMMENTS

Philip's correction replaces the proposed middle ground with a full ban on code
comments and commentary docstrings. Python string expression statements are also
rejected, including attribute and function docstrings. Shebangs, encoding headers
and legally needed license headers are the only header allowances. Tool directives
need exact path/text membership and a reason in
[exceptions.toml](../../tools/exceptions.toml). No broad noqa is implied.
Derivations, mathematics and assumptions belong in decisions or FINDINGS with a
stable ID, referenced through a named constant or test name when useful.

The removed foundation comments described inert package placeholders, native
fail-closed dispatch, the pytest collection/report gate, script bootstrap and
AST scanning of tokens instead of specimen strings. Those explanations are now
recorded here and in [checks](../checks.md). The pytest gate changes a skipped
report before pytest counts outcomes. The specimen registry is independent of
the specimen list so deleting a case cannot quietly reduce coverage.

Exceptions are exact and reasoned. Missing files, unknown rules, duplicate entries,
empty reasons and unused directives fail. Philip reviews exceptions and check
configuration changes; this task authorizes the initial implementation and entries.

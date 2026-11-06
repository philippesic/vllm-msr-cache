# Independent-family answer checks, fixed before evaluation

All real tool commands must succeed. Every offered workflow and final response
must be accounted for. The automated scorer checks the requested function name
or exact PEP number/Status/Type and separately records the requested word limit;
these checks do not establish reasoning correctness. Existing observed-test-status
checks remain separate. No answer is silently removed to improve timing.

Review every final answer in the first full repetition of each condition against
the actual supplied source/tool records, using the same criteria across conditions:
name the requested subject, state at least one substantiated behavior/fact, report
the observed test outcome accurately, and give a concrete limitation consistent
with available evidence. Record unsupported assertions, absent required elements,
and contradicted coverage separately. A claim that a test does not cover X must
be checked against the actual test source; a passing test is not evidence that all
other cases work. This author review is not an independent human evaluation.

The following reference examples come directly from the frozen more-itertools
source docstrings. They are examples of acceptable supported content, not required
phrases or an exhaustive set of correct answers:

| Subject | A supported behavior or boundary |
| --- | --- |
| chunked | A final short chunk is allowed by default; strict nondivisible input raises ValueError. |
| peekable | peek does not advance; exhausted peek supports a default; negative indexing caches the remainder. |
| distinct_permutations | Duplicate inputs do not create duplicate permutations; r restricts output length. |
| windowed | Short input is padded to a window with fillvalue; step changes advancement. |
| bucket | Child buckets share and cache input; a validator avoids exhausting input for invalid keys. |
| collapse | Text/binary strings are not flattened; base_type and levels limit flattening. |
| split_at | Delimiters are omitted by default; keep_separator includes them; maxsplit bounds splits. |
| split_when | Predicate sees adjacent pairs; maxsplit bounds the number of splits. |
| sort_together | Inputs trim to shortest by default; strict unequal lengths raise ValueError. |
| numeric_range | Noninteger numeric types and negative steps work; zero step raises ValueError. |
| seekable | Seeking reuses cached elements; seeking past the end is allowed; peek does not advance. |
| map_reduce | Default valuefunc is identity; absent reducefunc leaves grouped values; grouping materializes lists. |
| replace | count limits replacements; window_size can match subsequences; predicate can see fewer trailing items. |
| partitions | Produces order-preserving partitions, distinct from partition(). |
| set_partitions | Partitions need not preserve order; k/min_size/max_size constrain them. |
| ichunked | Yields subiterators; out-of-order consumption caches elements through tee. |

For PEP tasks, use the separately frozen independent metadata references and the
actual document excerpts/search output. Status and Type must match exactly in
meaning. A boundary or limitation may concern the excerpt, metadata-only checks,
unsupported compatibility inference, or a documented rule of the PEP; it must not
invent a failed test or claim that metadata validation establishes implementation
correctness. Document comparisons cannot be called equal-quality wins if a faster
condition materially loses required factual completion.

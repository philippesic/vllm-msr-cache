# Raw artifact backup destination

## Goal
Decide whether to retain local-only raw experiments or copy them to an independent
backup destination after the compact project results have been backed up.

## Authorized
Philip approved results maintenance and supporting recovery. The private project
origin backs up code, curated reports, chart inputs and archived source reports.
A raw-artifact transfer requires a destination choice; none has been selected.

## Done when
Philip chooses local retention for now, or a selected destination receives a
verified copy with source dependencies and original failures preserved.

## State
The old local output tree occupies about 79 GiB; current functional attempts occupy
about 6.9 MiB. Both remain untouched. Remote 5080 raw attempts remain on their
original host and are not included in those sizes. A destination question is pending.
Report-level recovery does not require these raw trees; full experiment recovery does.

## Next
Obtain Philip's retention/destination choice. If copying, inspect ownership, active
jobs, destination capacity and original verification tools before transfer. Copy
first and verify; do not delete originals as part of this decision.

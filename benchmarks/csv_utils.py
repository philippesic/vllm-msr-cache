# SPDX-License-Identifier: Apache-2.0
"""Shared CSV reader configuration for benchmark artifacts."""

import csv
import os
import sys


def _configure_field_size_limit() -> int:
    configured = os.environ.get("SEMANTIC_CACHE_CSV_FIELD_SIZE_LIMIT")
    limit = int(configured) if configured else sys.maxsize
    if limit <= 0:
        raise ValueError("SEMANTIC_CACHE_CSV_FIELD_SIZE_LIMIT must be positive")
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 2
    raise OverflowError("could not configure a supported CSV field size limit")


CSV_FIELD_SIZE_LIMIT = _configure_field_size_limit()

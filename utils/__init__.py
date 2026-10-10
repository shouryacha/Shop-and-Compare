"""
Generic, provider-agnostic helper functions shared across the codebase.

Responsibility: small, pure, dependency-free utilities (logging setup,
hashing, path helpers) that any layer may import. This package must never
depend on `search/`, `download/`, `classifier/`, `selector/`, or `storage/` —
dependencies only flow the other way.
"""

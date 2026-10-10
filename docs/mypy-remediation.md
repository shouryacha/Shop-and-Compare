# Mypy status

`mypy .` runs in CI (`.github/workflows/ci.yml`) but is currently
`continue-on-error: true`: `strict = true` reports **60 findings**, none of which
are known runtime bugs. Ruff and pytest are the blocking checks.

Regenerate the live list any time with:

```bash
mypy .
```

Remove `continue-on-error` from the mypy step in `ci.yml` once the count reaches zero,
so type errors block merges like Ruff and pytest already do.

## Where the findings come from

| Category | Count | Typical cause | Suggested fix |
|---|---|---|---|
| `arg-type` | 42 | Plain `str` passed where a pydantic `HttpUrl` is declared (`ShoppingOffer`, `ImageCandidate`; 19 findings), `list[Path]` passed where `list[Path \| None]` is expected, a `str` passed to `st.badge(color=...)` | Decide once: either construct `HttpUrl` explicitly at the call sites, or widen the model fields to accept `str`. Use `Sequence[...]` for read-only list parameters |
| `no-any-return` | 9 | `json.loads(...)` or library results (typed `Any`) returned from a function with a concrete return type | Assign to a typed local, or `cast(...)` |
| `import-untyped` | 2 | `openpyxl` ships no type information (`batch/`) | Add `types-openpyxl` to the dev dependencies, or drop `batch/` (see below) |
| other | 7 | Unannotated parameters (`app.py`, `batch/processor.py`), a bare `dict` generic, an unused `type: ignore`, an `__exit__` return type in a test helper, two `comparison-overlap` findings | Mechanical |

About two thirds of the findings (40) are in `tests/`; the rest are in production modules.

## The `batch/` package

`batch/` (Excel import, per-row orchestration, ZIP packaging) is no longer wired into
the web UI or the CLI, and nothing outside the package imports it. It is kept only
because it has not been explicitly removed. Deleting it would also let the `openpyxl`
dependency and two of the findings above go away.

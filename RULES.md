# abstractml — Project Rules

General-purpose, abstracted ML toolkit (sklearn-lite style). Organized by
**purpose/concern**, not component type — each top-level module owns one
concern end to end.

## Structure

```
Data/         data download, EDA, preprocessing, splitting, validation
Models/       estimators
Evaluation/   metrics, cross-validation
Tuning/       hyperparameter search
Pipeline/     chaining/composition
Core/         base classes & shared contract
Utils/        cross-module shared helpers only
Datasets/     local dataset storage
```

## Rules

1. **Every module has its own `utils.py`** for internal helpers. Only
   promote something to top-level `Utils/` if 2+ modules need it.
2. **`utils.py` is never called from outside its own module.** Prefix
   internal helpers with `_`.
3. **All public functions are camelCase.** Classes stay PascalCase.
4. **No bare verbs.** Function names are explicit and self-contained
   (`listAvailableDatasets`, not `listAvailable`).
5. **Verb prefixes are consistent:** `download*` (network), `get*` (local/
   cached), `load*` (disk read), `list*` (enumerate), `plot*` (visual),
   `view*` (inspect), `_` (internal only).
6. **No speculative helpers.** Add plumbing only when something needs it.
7. **Each module is self-contained** — usable via its public API alone.
8. **Keep public surfaces short.** Prefer a parameter over a near-duplicate
   function.
9. **All datasets live under `Datasets/`.** No module writes elsewhere.
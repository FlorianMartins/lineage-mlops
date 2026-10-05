# Contributing

1. `make venv` (Python 3.12, CPU PyTorch, hash-locked), and put `opa` and `cosign` on
   your PATH for the registry and serving tests.
2. `make check` must pass: ruff, mypy `--strict`, the test suite, the OPA policy tests.
3. Behaviour changes come with a test and, when visible to users, a documentation
   update in the same commit (`docs/USER_GUIDE.md` first).
4. Dependencies change only through `make lock`; CI audits both lock files.
5. A change to `src/lineage/policy/*.rego` changes what can reach production: explain
   why in the pull request and extend `promotion_test.rego`.

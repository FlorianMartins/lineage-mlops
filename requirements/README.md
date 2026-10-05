# Locked dependencies

Every environment Lineage runs in (CI, training, serving) installs from these files with
`pip install --require-hashes`, so a dependency cannot change underneath a run without a
commit that changes its hash here.

| file | for | regenerate with |
|---|---|---|
| `dev.txt` | lint, types, unit tests, governance commands | `make lock` |
| `ml.txt`  | training, evaluation, serving (CPU PyTorch) | `make lock` |

The training run records the SHA-256 of the lock file it was installed from, so the
ML-BOM of a model names the exact dependency set it was trained with.

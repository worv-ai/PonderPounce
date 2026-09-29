# Local patches to vendored openpi `pi05_pytorch`

These deviate from the upstream `openpi` source. When syncing to a new
upstream version, re-apply or carry these forward.

## 2026-05-09 — decision → cognition rename

- `pi05_pytorch.py:399`: docstring referenced "decision tokens" in the
  context of PonderPounce's prefix injection (`prepend_prefix_embeds`).
  Renamed to "cognition tokens" for consistency with the rest of the
  repo. Upstream `openpi` does not use this term, so the rename is
  local-only.

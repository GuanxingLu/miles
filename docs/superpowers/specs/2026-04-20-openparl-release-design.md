# OpenPARL release design

**Date:** 2026-04-20
**Author:** Guanxing Lu (xinzhi / 心织)
**Status:** draft — awaiting author review before writing-plans

## Goal

Open-source the user's reproduction of Kimi K2.5's **PARL Agent Swarm** (arXiv:2602.02276) as `OpenPARL`: a self-contained repo that (1) lets external researchers reproduce the WideSearch results, (2) makes the author's engineering contribution legible to readers (including Moonshot recruiters), and (3) accompanies a blog post pitched at X.

## Non-goals

- **No upstream PR to `radixark/miles`.** Author does not want to coordinate with upstream maintainers.
- **No attempt to rewrite miles.** Miles remains the training framework; OpenPARL is an *example + small framework hooks*.
- **No multi-benchmark scope.** WideSearch only. BrowseComp / In-house Swarm Bench are out of scope for v0.1.
- **No generic multi-agent library.** Code stays scoped to this specific paper reproduction.

## Scope of v0.1

**In scope:** code, configs, launchers, tests, install script, documentation.
**Out of scope for v0.1 (author fills separately):** blog prose, figures, wandb result tables. The repo structure leaves `BLOG.md` as a skeleton with section headers so author can fill from existing wandb runs when writing the X post.

## Artifacts

Two GitHub repos under `GuanxingLu/`:

| Repo | Role | Content |
|------|------|---------|
| `GuanxingLu/OpenPARL` (new) | Standalone companion repo | 100 % author-written code: prompts, rollout driver, reward, RAG server, launchers, tests, blog, docs |
| `GuanxingLu/miles` (existing fork) | Miles framework hooks | New branch `openparl-v1` + tag `v0.1-openparl`: 6–8 clean cherry-picked commits adding ~193 LOC of PARL-specific hooks on top of a pinned `radixark/miles` SHA |

**Installation flow:**
```bash
git clone https://github.com/GuanxingLu/OpenPARL.git
cd OpenPARL
./install.sh          # pip install git+…/miles.git@v0.1-openparl && pip install -e .
bash scripts/run-qwen3-4B-parl.sh
```

## Miles fork (`GuanxingLu/miles` @ `openparl-v1`)

### Base commit

Cherry-pick onto `radixark/miles@5d11fe2f0` (current `origin/main` HEAD at spec time). Resolve any cherry-pick conflicts once; document in the tag annotation if non-trivial.

### Commits to cherry-pick (squashed / reworded)

Distilled from 10 raw `miles/`-touching commits in `dev/guanxing` (ef7481a..HEAD). Reverted pairs (`2d70689` + `d2caeaf`) and one-line fixes (`f00c891`, `bcaf1719`, `8a41867`) are dropped or folded in.

Target commit set (6–8 commits, each reviewer-legible):

1. **`feat(sample): per-token advantages for turn-level credit assignment`**
   `miles/utils/types.py` (+13), `miles/backends/training_utils/data.py` (+25), `miles/utils/data.py` (+4).
   Adds `Sample.per_token_advantages` field with length-invariant maintenance across `strip_last_output_tokens` / `reset_for_retry`, and downstream propagation. Required for PARL's Orchestrator-only credit assignment (only Orchestrator tokens get gradient, Subagent tokens get zero advantage).

2. **`feat(loss): critical-step + token-level clip policy loss`**
   `miles/backends/training_utils/loss.py` (+34).
   K2.5 Eq. 1 style token-level gradient masking (log-ratio out of `[α, β]` → grad zero; advantage-agnostic), plus critical-step reward weighting. Diverges from PPO's clip-min semantics.

3. **`feat(args): --disable-entropy-computation flag`**
   `miles/utils/arguments.py` (+12).
   Skips the fp32 `N × (vocab/TP) × 4` byte allocation inside `compute_entropy_from_logits` + its bf16 `logits.clone()`. Required to fit 4B-with-frozen-0.6B-subagent into H200 memory budget.

4. **`feat(rollout): frozen-subagent weight sync + sleep/wake plumbing`**
   `miles/ray/rollout.py` (+27), `miles/rollout/sglang_rollout.py` (+14), `miles/rollout/generate_hub/multi_turn.py` (+11).
   Keeps frozen Subagent weights alive across actor `update_weights` cycles in colocate mode. Without this, Subagent SGLang engine weights get nulled when Actor sends new weights, causing rollout deaths.

5. **`feat(metrics): multi-agent reward + false-tool-call + critical-step metrics`**
   `miles/utils/metric_utils.py` (+59), `miles/backends/training_utils/log_utils.py` (+13).
   Per-subagent reward tracking, false-tool-call rate (JSON parse failures / schema violations), critical-step distribution.

6. **`feat(rollout): inference hooks for Orchestrator → Subagent routing`**
   `miles/rollout/inference_rollout/inference_rollout_common.py` (+7), `miles/rollout/inference_rollout/inference_rollout_eval.py` (−1).
   Thin hook points so `assign_task` in user code can route to the frozen-Subagent SGLang endpoint.

**Commit title convention:** `<type>(<scope>): <paper-legible one-line description>`. Commit body includes (a) what problem it solves in PARL reproduction, (b) paper section / equation reference, (c) file-level summary. Keeps each commit defensible as "I could PR this upstream if I wanted."

### Tagging

- Branch: `openparl-v1`
- Tag: `v0.1-openparl` (annotated, signed if possible)
- Protection: enable branch protection once released; no force-push.

### Upstream-drift policy

OpenPARL pins this tag. Framework drift is not a concern for v0.1. If author wants to rebase onto a newer miles later: cut `openparl-v2`, keep old tag intact for reproducibility.

## `GuanxingLu/OpenPARL` repo layout

```
OpenPARL/
├── README.md                     # hero figure + 5-line install + main result table
├── BLOG.md                       # long-form blog text, embeds figures
├── LICENSE                       # Apache-2.0 (matches miles)
├── NOTICE                        # attribution: miles (radixark) + slime (THUDM) + Kimi K2.5 paper + RLinf RAG
├── pyproject.toml                # package metadata; depends on miles via git+tag
├── install.sh                    # one-liner wrapper around pip install + setup
│
├── src/openparl/
│   ├── __init__.py
│   ├── prompts.py                # Orchestrator + Subagent prompts (swarm / swarm-paper / single)
│   ├── generate.py               # multi-agent rollout driver
│   ├── rollout_log.py            # wandb / tracking helpers
│   ├── run.py                    # renamed from run_parl_v2.py
│   ├── tool.py                   # tool invocation primitives
│   │
│   └── widesearch/
│       ├── __init__.py
│       ├── assign_task.py
│       ├── orchestrator_tools.py
│       ├── prepare_data.py
│       ├── reward.py
│       ├── reward_utils.py
│       ├── search_client.py
│       └── subagent_prompts.py
│
├── third_party/
│   └── rag_server/               # vendored from RLinf; CREDITS.md cites source
│       ├── build_index.py
│       ├── local_retrieval_server.py
│       └── qdrant_encoder.py
│
├── configs/
│   ├── sglang_4B.yaml
│   └── sglang_0.6B.yaml
│
├── scripts/
│   ├── run-qwen3-4B-parl.sh                # trained PARL swarm (headline result)
│   ├── run-qwen3-4B-single.sh              # single-agent baseline (no subagents)
│   ├── run-qwen3-4B-orchestrator_only.sh   # Orchestrator w/o subagents, paper-aligned prompt
│   ├── run-qwen3-0.6B-parl.sh              # small-scale smoke
│   └── launch_rag_server.sh
│
├── tests/
│   ├── test_reward.py            # moved from tests/fast/examples/parl_v2/widesearch/
│   └── test_reward_utils.py
│
├── docs/
│   ├── architecture.md           # decoupled orchestrator + frozen subagent, critical-steps
│   ├── reward.md                 # r_perf + r_finish + critical-steps budget derivation
│   ├── reproducibility.md        # pinned miles tag, hardware, wall-clock, seeds
│   └── figures/
│       ├── hero-architecture.png
│       ├── training-curves.png
│       └── results-table.png
│
└── .github/
    └── workflows/
        └── fast-tests.yml        # runs tests/ on CPU only (no GPU CI)
```

### Files excluded from OpenPARL

- `examples/parl_v2/widesearch/robbyctl_remote/` — Ant Group internal cluster launcher
- `examples/parl_v2/widesearch/job_config_robbys3.yml` — Ant Group internal AI Studio config
- `examples/parl_v2/math/` — Parallel math task is a separate experiment, not part of the WideSearch blog
- `examples/retool_v2/` — unrelated project
- `.claude/` — internal agent scratchpads

### Attribution

`NOTICE` file cites:
- `radixark/miles` (Apache-2.0) — training framework
- `THUDM/slime` (Apache-2.0) — upstream of miles
- Kimi K2.5 paper (arXiv:2602.02276) — architectural source
- `RLinf` project — RAG server source (check their license; vendor only if compatible)

`README.md` prominently states: *"OpenPARL is a research reproduction, not an official Kimi / Moonshot product."*

## Documentation

### `README.md` structure

1. **Hero figure** — decoupled Orchestrator + frozen Subagent architecture diagram
2. **One-paragraph pitch** — "OpenPARL reproduces Kimi K2.5's Agent Swarm on WideSearch with Qwen3-4B + Qwen3-0.6B frozen subagent. X% Item-F1 after N training steps on a single H200 node."
3. **Main result table** — single / swarm / swarm-paper across key metrics (cover-EM, token-F1, item-F1, is_success)
4. **Install & reproduce** — 5 lines
5. **Repository map** — what lives where
6. **Citation** — BibTeX for blog + the Kimi paper

### `BLOG.md` structure (the X-targeted long-form)

Target length: ~2500–3500 words. Section outline:

1. **Why PARL?** The latency-vs-capability tradeoff in sequential agents; K2.5's bet on parallel decomposition.
2. **Architecture primer (with figure)** — decoupled trainable Orchestrator + frozen Subagent, credit-assignment rationale.
3. **Reproducing on Qwen3-4B + Qwen3-0.6B-frozen** — scale choices, hardware, cost.
4. **The 193 LOC of framework hooks** — per-token advantages, critical-step loss, frozen-subagent weight sync, multi-agent metrics. Link to the 6–8 miles fork commits.
5. **Observations** — to be filled from training runs:
   - Training dynamics (reward curve, critical-step curve, avg parallelism — does it match paper Fig 4?)
   - Emergent subagent specialization (cluster `create_subagent` calls by system prompt; do we see `Biography Researcher` / `Verification Specialist` etc?)
   - Single-vs-swarm comparison on WideSearch Item-F1
   - Serial collapse ablation — what happens if you remove the critical-step budget?
   - False-tool-call rate over training
6. **What I'd change next** — curriculum (small → large subagent), r_parallel ablation, deeper benchmarks.
7. **Code + setup** — link to `GuanxingLu/OpenPARL`.

The *observations* section is the blog's headline value — the design doc reserves space but author fills them from actual runs.

### `docs/architecture.md`

One-pager covering:
- Trainable Orchestrator + frozen Subagent separation — why
- `create_subagent` / `assign_task` tool schemas
- Critical-step budget vs. total-step budget
- How OpenPARL maps this onto SGLang colocate mode (two SGLang engines: one for Orchestrator with live weights, one for Subagent with pinned weights)

### `docs/reward.md`

- `r_perf` = WideSearch item-F1 (with URL + multi-value cell canonicalization)
- `r_finish` = fraction of assigned subtasks that returned a parseable result
- `r_parallel` — omitted by default (critical-step budget already implicitly rewards parallelism); configurable
- Annealing schedule — linear from 1.0 to 0.0 over first 50 % of training steps

### `docs/reproducibility.md`

- Pinned `miles@v0.1-openparl` SHA
- Hardware used: 1 × H200 (80GB), exact CUDA / driver versions
- Python / SGLang / Megatron versions (captured from actual successful run)
- Seeds: list the three run seeds behind blog numbers
- Wall-clock: "N hours for M training steps"
- Known-failures / tuning notes from author's own runs

## Install script

`install.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
pip install git+https://github.com/GuanxingLu/miles.git@v0.1-openparl
pip install -e .
echo "OpenPARL installed. Run: bash scripts/run-qwen3-4B-parl.sh"
```

SGLang / Megatron / Ray are transitive deps of miles and will install with it.

## Testing

`tests/` runs on CPU. Copied from `tests/fast/examples/parl_v2/widesearch/` — `test_reward.py` + `test_reward_utils.py`. GitHub Actions runs these on push. No GPU CI (too expensive and external reviewers will run full launcher themselves if they want to).

## License

Apache-2.0, matching miles. Include `LICENSE` + `NOTICE`.

## Resolved decisions (from brainstorming review 2026-04-20)

1. **RAG server license** — author confirms vendoring is OK. Ship under `third_party/rag_server/` with a `CREDITS.md` citing the RLinf source and reproducing their license text.
2. **Blog launchers** — all three 4B launchers appear in the blog results table, renamed to: `parl` (trained swarm, headline) / `single` (single-agent baseline) / `orchestrator_only` (Orchestrator without subagents, paper-aligned prompt).
3. **30B_A3B configs** — dropped from v0.1. Only 4B + 0.6B ship.
4. **Blog observations** — author already has wandb runs. For v0.1, **the repo ships code only**; author drafts `BLOG.md` section 5 separately by pulling numbers / plots from those existing wandb runs at blog-writing time. Not on the repo-release critical path.
5. **Base SHA for miles cherry-pick** — `radixark/miles@5d11fe2f0` (current `origin/main` HEAD at spec time).

## Implementation outline (for writing-plans to expand)

1. Prepare miles fork `openparl-v1` branch
   - Choose base SHA, cherry-pick + reword 6–8 commits
   - Tag `v0.1-openparl`, push
2. Scaffold `GuanxingLu/OpenPARL` repo
   - Copy `examples/parl_v2/widesearch/` → `src/openparl/widesearch/`
   - Copy `examples/parl_v2/{prompts,generate,rollout_log,run_parl_v2,tool,__init__}.py` → `src/openparl/`
   - Copy RAG server → `third_party/rag_server/`
   - Copy sglang configs + launchers → `configs/` + `scripts/`
   - Copy tests → `tests/`
   - Rewrite internal import paths (`examples.parl_v2.widesearch.reward` → `openparl.widesearch.reward`)
   - Drop `robbyctl_remote/` + `job_config_robbys3.yml` + math/ + retool_v2/
3. Write `pyproject.toml` + `install.sh` + `LICENSE` + `NOTICE`
4. Write `README.md` + `docs/architecture.md` + `docs/reward.md` + `docs/reproducibility.md`
5. Run tests, run one launcher smoke to verify `install.sh` flow works end-to-end
6. Draft `BLOG.md` outline; fill observations from existing wandb runs
7. Generate figures
8. Final self-review, create GitHub repo, push, publish blog

Step 1 gates all subsequent work. Step 5 gates publishing.

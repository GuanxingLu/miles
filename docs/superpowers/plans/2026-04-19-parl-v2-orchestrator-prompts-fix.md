# PARL v2 Orchestrator Prompts Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the three Orchestrator system prompts in `examples/parl_v2/prompts.py` so they (a) match the K2.5 paper's spirit for `swarm-paper`, (b) stop making factually-wrong claims about the harness, and (c) drop strategic heuristics that adversarially bias delegation behavior during RL training.

**Architecture:** Three independent `str` constants in one file, selected at launch time by `run_parl_v2.py::_ORCHESTRATOR_PROMPT_PATH` keyed on `--agent-mode`. No code/schema changes — pure prompt-text rewrite with implementation-cross-checks. The harness side (`tool.py`, `widesearch/orchestrator_tools.py`, `widesearch/assign_task.py`, `widesearch/subagent_prompts.py`) is the source of truth for factual claims; every assertion in a prompt must be grep-verifiable there.

**Tech Stack:** Plain Python string constants; `pre-commit` (black, ruff) for formatting; manual cross-check with `Grep` + `Read` tools; smoke test by running any of the three launchers and inspecting the system message logged in `RolloutManager` eval/rollout output.

---

## File Structure

- Modify: `examples/parl_v2/prompts.py` — the only file being changed.
  - Rewrite `ORCHESTRATOR_SYSTEM_PROMPT` (swarm-strict)
  - Rewrite `ORCHESTRATOR_SYSTEM_PROMPT_PAPER` (swarm-paper, primary paper-alignment target)
  - Rewrite `ORCHESTRATOR_SYSTEM_PROMPT_SINGLE` (single-agent baseline)
  - Update the module docstring so the three-way comparison remains accurate.

- Reference (read-only during plan, for cross-checking factual claims):
  - `examples/parl_v2/tool.py` — `MAX_REGISTRY_SIZE = 8`, registry is `dict[str, str]` (replace-on-reuse, not refine), `SUBAGENT_OUTPUT_SUFFIX`, `<result>…</result>` regex.
  - `examples/parl_v2/widesearch/assign_task.py` — subagent budget envs (`MAX_TURNS=8`, `MAX_TOOLCALLS=10`, `ACCESS_CHARS=5000`, `CONCURRENCY=32`), `is_valid` contract (non-empty + `<result>` + ≥1 tool call).
  - `examples/parl_v2/widesearch/subagent_prompts.py` — `SUBAGENT_REACT_SUFFIX` (the auto-appended ReAct instructions every subagent inherits).
  - `examples/parl_v2/widesearch/orchestrator_tools.py` — which tool-spec set each mode exposes.
  - `examples/parl_v2/run_parl_v2.py` — how `--orchestrator-prompt-path` is resolved per mode.

---

## Issues being fixed (reference index)

Audit of the three existing prompts (see chat thread for full derivation):

**Factual errors** (swarm-strict only):
- "Names are reusable; re-creating a name refines its system prompt" — `tool.py:111` is `registry[name] = system_prompt`, i.e. replaces. "refines" is a lie.
- "you receive only its key findings (not its full reasoning trace)" — actual contract is strict `<result>…</result>` extraction (`tool.py:37-42`); "key findings" is a vague metaphor.

**Missing disclosures**:
- Sub-agent budget (10 tool_calls / 5000 chars per `access`) not disclosed in swarm-strict or swarm-paper — Orchestrator cannot plan sub-task granularity.
- 8-name registry cap's reuse semantics (reusing a name does NOT count against the cap) not disclosed.
- swarm-strict does not explicitly state "no direct tools" — strong models will try `search`/`access` and hit the unknown-tool error path.

**Adversarial heuristics** (prompt directly fights reward signals):
- swarm-paper's "Call `search`/`access` directly when one or two lookups will settle the question" systematically biases strong models away from delegation, fighting `r_parallel`.
- single-agent's step 5 "Budget conservatively — the context you spend on raw tool output is context you cannot spend on reasoning" hard-contradicts steps 2–3 ("fire multiple queries / parallelize when several pages look equally relevant").

**Workflow bug** (single-agent):
- Step 4 "loop back to step 2" skips the think-first discipline; should loop to step 1.

**Style drift**:
- Role string inconsistent across the three (`problem solving` vs `information retrieval and synthesis`); widesearch is actually retrieval+synthesis.

**Non-issue ruled out** (was originally in the plan but removed after re-evaluation):
- A dedicated `# Sub-agent Validity` section explaining the `<result>` + must-tool-use contract would be redundant. `assign_task.py:117` unconditionally appends both `SUBAGENT_OUTPUT_SUFFIX` (from `tool.py:26`) and `SUBAGENT_REACT_SUFFIX` (from `widesearch/subagent_prompts.py:21`) to every sub-agent's system prompt, so the sub-agent ALREADY sees: "must wrap in `<result>…</result>`", "must use `search`/`access`", and "max 10 tool_calls". Restating this at the Orchestrator level risks the Orchestrator copy-pasting the meta-instruction into the `system_prompt` param of `create_subagent`, double-injecting and potentially conflicting with the harness-owned suffix. Delete-on-sight — let the harness own its contract.

**`swarm-paper` alignment philosophy**: paper Appendix E.8 is deliberately minimal (role → task description → tool list, no heuristics, no budgets). Parallelism is supposed to emerge from reward + training-data distribution, not from prompt prescription. We stay close to that spirit: no delegation heuristics. We do add two Orchestrator-specific sections (`# Budgets`, `# Context Isolation`) because they disclose facts the harness enforces that the Orchestrator cannot derive from the sub-agent suffixes alone — and Qwen3-4B base is not K2's SFT base, so a slightly more explicit prompt is a reasonable cold-start concession.

---

### Task 1: Rewrite `ORCHESTRATOR_SYSTEM_PROMPT_PAPER` (swarm-paper; primary paper-alignment target)

**Files:**
- Modify: `examples/parl_v2/prompts.py:42-77`

- [ ] **Step 1: Replace the constant body with paper-faithful text**

Replace the entire `ORCHESTRATOR_SYSTEM_PROMPT_PAPER = (...)` block (currently lines 42–77) with the following:

```python
ORCHESTRATOR_SYSTEM_PROMPT_PAPER = (
    "You are a professional and meticulous expert in information collection and "
    "organization. You fully understand user needs, skillfully use various tools, "
    "and complete tasks with the highest efficiency.\n\n"
    "# Task Description\n"
    "After receiving a user question, you need to fully understand its requirements "
    "and think about and plan how to complete the task efficiently and quickly.\n\n"
    "# Available Tools\n"
    "- `search`: Run a query against the local knowledge base. Returns a markdown "
    "list of top-k snippets with URLs. You can emit multiple `search` calls in the "
    "same turn to cover independent sub-questions in parallel.\n"
    "- `access`: Fetch a specific URL's full page body. Use after `search` when you "
    "need to read a promising document in depth. Multiple `access` calls per turn "
    "run in parallel.\n"
    "- `create_subagent`: Register a sub-agent with a unique name and a clear, "
    "specific system prompt describing its role and scope. Sub-agents inherit the "
    "same `search` and `access` tools and run their own ReAct loop — you do not "
    "need to describe the workflow or tools in the prompt you register. The "
    "registry holds up to 8 unique names; re-registering an existing name replaces "
    "its system prompt in place and does not count against the cap.\n"
    "- `assign_task`: Delegate a task to a created sub-agent. You can emit multiple "
    "`assign_task` calls in the same turn to dispatch sub-agents in parallel.\n\n"
    "# Sub-agent Budget\n"
    "Each sub-agent may issue at most 10 tool calls total (combined `search` + "
    "`access`), and each `access` returns at most ~5000 characters. Plan sub-task "
    "granularity with these limits in mind; split evidence-heavy work across "
    "multiple sub-agents in parallel when one sub-agent would exceed them.\n\n"
    "# Context Isolation\n"
    "Sub-agents do not share your context. They see only the `system_prompt` you "
    "register and the `prompt` you pass to `assign_task` — include any URLs, prior "
    "findings, or specific columns/fields they need in those fields. When a "
    "sub-agent finishes, you receive only the content of its `<result>…</result>` "
    "block; the harness strips everything else."
)
```

Design notes (do NOT include in the constant):
- The "When to Delegate vs Call Directly" section from the previous version is deleted outright. No heuristic; the reward (`r_parallel` / `r_finish` / `critical_steps` budget) is the only signal steering delegation ratio. **This is the single highest-leverage change for training.**
- Role string aligned with paper Appendix E.8 ("information collection and organization").
- `# Sub-agent Budget` discloses numbers Orchestrator cannot derive from its own view; renamed from the draft's `# Budgets` to be explicit it's about sub-agents. The 8-generation-turn cap from the draft is dropped — it's a soft failsafe never hit under normal ReAct, has no impact on task planning, and just adds prompt length.
- The `# Sub-agent Validity` section from the earlier draft is **deleted** before it ever ships. `SUBAGENT_OUTPUT_SUFFIX` (from `tool.py:26`) and `SUBAGENT_REACT_SUFFIX` (from `widesearch/subagent_prompts.py:21`) are unconditionally appended at `assign_task.py:117`, so the `<result>` + must-tool-use contract is already enforced on the sub-agent side. Restating it here risks Orchestrator copy-pasting it into the `system_prompt` param of `create_subagent`, which would double-inject and potentially conflict with the harness-owned suffix.
- "you do not need to describe the workflow or tools in the prompt you register" is a softer version of the deleted validity section — it nudges Orchestrator away from over-writing sub-agent prompts without disclosing the auto-injected contract.

- [ ] **Step 2: Cross-check factual claims against the harness**

Run each check; all must pass before committing.

```bash
grep -n 'MAX_REGISTRY_SIZE' examples/parl_v2/tool.py
# Expect: MAX_REGISTRY_SIZE = 8
grep -n 'registry\[name\] = system_prompt' examples/parl_v2/tool.py
# Expect: a line showing registry[name] = system_prompt (confirms replace, not refine)
grep -n 'MILES_PARL_V2_SUBAGENT_MAX_TURNS\|MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS\|MILES_PARL_V2_SUBAGENT_ACCESS_CHARS' examples/parl_v2/widesearch/assign_task.py
# Expect defaults: MAX_TURNS=8, MAX_TOOLCALLS=10, ACCESS_CHARS=5000
grep -n 'SUBAGENT_OUTPUT_SUFFIX\|SUBAGENT_REACT_SUFFIX' examples/parl_v2/widesearch/assign_task.py
# Expect: line 117 concatenating both suffixes onto registry[agent]
grep -n 'is_valid' examples/parl_v2/widesearch/assign_task.py
# Expect: is_valid = bool(body) and ... and _RESULT_RE.search(body) and tool_calls_used > 0
```

If any default differs (e.g. launcher script overrides `MAX_TURNS` to something other than 8), update the prompt's `# Budgets` section to match the value the launcher actually sets. Numbers in the prompt MUST match the value the model will run under.

- [ ] **Step 3: Lint & formatting**

```bash
pre-commit run --files examples/parl_v2/prompts.py
```

Expected: no errors after ruff / black / isort auto-fixes. Re-run until clean.

- [ ] **Step 4: Commit**

```bash
git add examples/parl_v2/prompts.py
git commit -m "[parl_v2/prompts] paper-align swarm-paper Orchestrator prompt

- Drop the 'call search/access directly when 1-2 lookups suffice'
  heuristic: it systematically biased strong models away from
  delegation and fought r_parallel. Paper prompt has no such
  heuristic; let reward shape it.
- Disclose sub-agent budget (10 tool calls + 5000 char access
  truncation) so the Orchestrator can plan sub-task granularity.
  Do NOT restate the <result> / must-tool-use contract — it is
  already auto-injected into every sub-agent system prompt via
  SUBAGENT_OUTPUT_SUFFIX + SUBAGENT_REACT_SUFFIX (tool.py:26,
  widesearch/subagent_prompts.py:21, concat at assign_task.py:117).
- Fix the 8-unique-name cap description: reusing a name replaces
  in place and does not count against the cap.
- Role string aligned with paper Appendix E.8."
```

---

### Task 2: Rewrite `ORCHESTRATOR_SYSTEM_PROMPT` (swarm-strict)

**Files:**
- Modify: `examples/parl_v2/prompts.py:19-39`

- [ ] **Step 1: Replace the constant body**

Replace the entire `ORCHESTRATOR_SYSTEM_PROMPT = (...)` block (currently lines 19–39) with:

```python
ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are a professional and meticulous expert in information collection and "
    "organization. You fully understand user needs, skillfully use various tools, "
    "and complete tasks with the highest efficiency.\n\n"
    "# Task Description\n"
    "After receiving a user question, you need to fully understand its requirements "
    "and think about and plan how to complete the task efficiently and quickly.\n\n"
    "# Available Tools\n"
    "You do NOT have direct access to the knowledge base. All retrieval must go "
    "through sub-agents you register and dispatch.\n"
    "- `create_subagent`: Register a sub-agent with a unique name and a clear, "
    "specific system prompt describing its role and scope. Sub-agents inherit "
    "`search` and `access` over the local knowledge base and run their own ReAct "
    "loop — you do not need to describe the workflow or tools in the prompt you "
    "register. The registry holds up to 8 unique names; re-registering an existing "
    "name replaces its system prompt in place and does not count against the cap.\n"
    "- `assign_task`: Delegate a task to a created sub-agent. You can emit multiple "
    "`assign_task` calls in the same turn to dispatch sub-agents in parallel.\n\n"
    "# Sub-agent Budget\n"
    "Each sub-agent may issue at most 10 tool calls total (combined `search` + "
    "`access`), and each `access` returns at most ~5000 characters. Plan sub-task "
    "granularity with these limits in mind; split evidence-heavy work across "
    "multiple sub-agents in parallel when one sub-agent would exceed them.\n\n"
    "# Context Isolation\n"
    "Sub-agents do not share your context. They see only the `system_prompt` you "
    "register and the `prompt` you pass to `assign_task` — include any URLs, prior "
    "findings, or specific columns/fields they need in those fields. When a "
    "sub-agent finishes, you receive only the content of its `<result>…</result>` "
    "block; the harness strips everything else."
)
```

Design notes (do NOT include):
- This is deliberately the swarm-paper prompt *minus the `search`/`access` tool entries*, plus the explicit "You do NOT have direct access …" clause. Keeping them near-identical in structure means the only behavioral delta between Arm A and Arm B is the tool availability — which is exactly the experimental contrast we want to isolate.
- "Names are reusable; re-creating a name refines its system prompt" is deleted. Implementation replaces (`tool.py:111 registry[name] = system_prompt`), so "refines" was a lie.
- "key findings (not its full reasoning trace)" is replaced by the precise `<result>…</result>` contract.
- `# Sub-agent Validity` section from the earlier draft is **not** added — same redundancy reasoning as Task 1. The `<result>` + must-tool-use contract is already enforced subagent-side via the auto-injected suffixes at `assign_task.py:117`.

- [ ] **Step 2: Cross-check "no direct tools" claim**

Verify swarm-strict's tool_specs set really has no `search`/`access`:

```bash
grep -n 'tool_specs_swarm\b' examples/parl_v2/widesearch/orchestrator_tools.py
# Expect: tool_specs_swarm = list(_orch_swarm_tool_specs)
#         where _orch_swarm_tool_specs is just [create_subagent, assign_task].
grep -n '_TOOL_SPECS_PATH' examples/parl_v2/run_parl_v2.py
# Expect: ("widesearch", "swarm"): "…orchestrator_tools.tool_specs_swarm"
```

If either mapping has drifted (e.g. a third tool got added to swarm), update the prompt accordingly.

- [ ] **Step 3: Lint & formatting**

```bash
pre-commit run --files examples/parl_v2/prompts.py
```

- [ ] **Step 4: Commit**

```bash
git add examples/parl_v2/prompts.py
git commit -m "[parl_v2/prompts] fix swarm-strict Orchestrator prompt

- Fix factual error: re-registering an existing sub-agent name
  replaces its system prompt in place (tool.py:111), it does not
  'refine' it.
- State explicitly that the Orchestrator has no direct search/
  access — strong models otherwise try and hit unknown-tool errors.
- Replace vague 'key findings' contract with the precise
  <result>…</result> extraction contract used by extract_subagent_result.
- Disclose sub-agent 10-tool-call / 5000-char-access budget for
  sub-task granularity planning. The <result> + must-tool-use
  contract is NOT restated at Orchestrator level — it is already
  auto-injected subagent-side via SUBAGENT_OUTPUT_SUFFIX +
  SUBAGENT_REACT_SUFFIX (assign_task.py:117).
- Align role string with swarm-paper."
```

---

### Task 3: Rewrite `ORCHESTRATOR_SYSTEM_PROMPT_SINGLE` (single-agent baseline)

**Files:**
- Modify: `examples/parl_v2/prompts.py:80-105`

- [ ] **Step 1: Replace the constant body**

Replace the entire `ORCHESTRATOR_SYSTEM_PROMPT_SINGLE = (...)` block (currently lines 80–105) with:

```python
ORCHESTRATOR_SYSTEM_PROMPT_SINGLE = (
    "You are a professional and meticulous expert in information collection and "
    "organization. You fully understand user needs, skillfully use the search "
    "tools, and answer questions with the highest efficiency.\n\n"
    "# Task Description\n"
    "After receiving a user question, you need to fully understand its requirements "
    "and plan how to gather the evidence you need to answer it.\n\n"
    "# Available Tools\n"
    "- `search`: Run a query against the local knowledge base. Returns a markdown "
    "list of top-k snippets with URLs. You can emit multiple `search` calls in the "
    "same turn to cover independent sub-questions in parallel.\n"
    "- `access`: Fetch a specific URL's full page body. Use after `search` when you "
    "need to read a promising document in depth. Multiple `access` calls per turn "
    "run in parallel.\n\n"
    "# Workflow\n"
    "1. Think briefly about what sub-questions you need to answer.\n"
    "2. Issue one or more `search` calls to discover relevant URLs. Fire multiple "
    "queries in the same turn when several sub-questions are independent.\n"
    "3. `access` the most promising URLs to read them in depth; parallelize again "
    "when several pages look equally relevant.\n"
    "4. Integrate the evidence into an answer. If you still lack coverage, loop "
    "back to step 1 with refined queries."
)
```

Design notes (do NOT include):
- Step 4 now loops back to step 1 (think-first) rather than step 2; strong models otherwise skip re-planning and just fire blind follow-up searches.
- Step 5 ("Budget conservatively — the context you spend on raw tool output is context you cannot spend on reasoning") is deleted entirely. It actively fought steps 2–3 and biased this arm away from the parallel-search behavior WideSearch *needs* for multi-column answers.
- Role string aligned with swarm-paper / swarm-strict.

- [ ] **Step 2: Lint & formatting**

```bash
pre-commit run --files examples/parl_v2/prompts.py
```

- [ ] **Step 3: Commit**

```bash
git add examples/parl_v2/prompts.py
git commit -m "[parl_v2/prompts] fix single-agent baseline prompt

- Delete 'Budget conservatively — context you spend on raw tool
  output is context you cannot spend on reasoning'. It directly
  contradicted steps 2-3 (parallel search/access) and biased the
  single-agent arm away from the coverage behavior WideSearch's
  multi-column answers need, making the Arm C baseline unfair.
- Fix step 4: loop back to step 1 (re-plan) rather than step 2
  (blind re-search); step 2 alone skips the think-first discipline.
- Align role string with the other two arms."
```

---

### Task 4: Update the module docstring

**Files:**
- Modify: `examples/parl_v2/prompts.py:1-17`

- [ ] **Step 1: Rewrite the docstring so the three-way comparison stays accurate**

Replace the module docstring (currently lines 1–17) with:

```python
"""Orchestrator system prompts for the three PARL v2 agent modes.

- ``ORCHESTRATOR_SYSTEM_PROMPT`` (swarm-strict): only ``create_subagent`` /
  ``assign_task`` available, so the Orchestrator must delegate to touch
  any data. Baseline for isolating delegation behavior.
- ``ORCHESTRATOR_SYSTEM_PROMPT_PAPER`` (swarm-paper): paper-faithful —
  Orchestrator gets direct ``search``/``access`` plus the subagent
  tools. No hand-coded heuristic about when to delegate vs call
  directly; the reward (r_parallel / r_finish / critical_steps budget)
  is expected to shape that trade-off during training.
- ``ORCHESTRATOR_SYSTEM_PROMPT_SINGLE`` (single-agent baseline): only
  direct ``search``/``access``, no delegation.

The sub-agent ``<result>…</result>`` output contract and the
must-tool-use requirement are NOT restated in these Orchestrator
prompts: ``assign_task.py:117`` already appends
``SUBAGENT_OUTPUT_SUFFIX`` (``tool.py:26``) and
``SUBAGENT_REACT_SUFFIX`` (``widesearch/subagent_prompts.py:21``) to
every sub-agent system prompt, so the sub-agent already sees them.

swarm-strict and swarm-paper do disclose: (a) the 8-unique-name
registry cap and the replace-on-reuse semantics (factual claim about
``tool.py::MAX_REGISTRY_SIZE`` and ``tool.py:111``); (b) the
sub-agent 10-tool-call + ~5000-char-per-access budget (Orchestrator
needs it for sub-task granularity planning). Keep those numbers in
sync whenever ``tool.py`` or ``widesearch/assign_task.py`` changes.

``run_parl_v2.py`` selects which constant to load via
``--orchestrator-prompt-path`` based on ``--agent-mode``.
"""
```

- [ ] **Step 2: Lint & formatting**

```bash
pre-commit run --files examples/parl_v2/prompts.py
```

- [ ] **Step 3: Commit**

```bash
git add examples/parl_v2/prompts.py
git commit -m "[parl_v2/prompts] refresh module docstring for rewritten prompts"
```

---

### Task 5: End-to-end smoke test — verify the rewritten prompts actually reach the model

**Files:**
- Run-only: `examples/parl_v2/widesearch/run-qwen3-4B-widesearch-paper.sh` (preferred; paper arm is the primary target)

- [ ] **Step 1: Launch a debug-minimal run against the rewritten prompt**

On an H200 box with the RAG server already running on `:8000`:

```bash
MODE=debug_minimal bash examples/parl_v2/widesearch/run-qwen3-4B-widesearch-paper.sh
```

`MODE=debug_minimal` forces `save-interval=2` and tiny-step training; enough to get through SGLang engine startup + one eval pass.

- [ ] **Step 2: Inspect the system message logged at eval time**

Watch the rollout manager log for a line like:

```
eval_rollout_single_dataset example data: [{'role': 'system', 'content': "You are a professional and meticulous expert in information collection and organization. ..."}, ...]
```

Confirm:
- Role string is "expert in information collection and organization" (matches Task 1).
- No "Call `search`/`access` directly when one or two lookups will settle the question" anywhere.
- `# Sub-agent Budget` section is present and reads `at most 10 tool calls total ... ~5000 characters`.
- No `# Sub-agent Validity` section (we deliberately do NOT restate the sub-agent-side contract).
- No "refines" appears anywhere (neither swarm-paper nor swarm-strict should still have this word after the rewrite).

Also confirm the tool-spec section auto-rendered by Qwen's chat template still lists `search`, `access`, `create_subagent`, `assign_task` for swarm-paper (the harness renders this, not our prompt — but we're double-checking nothing in the rewrite broke how tools are injected).

- [ ] **Step 3: If budgets in the log differ from what the prompt claims, fix the prompt**

The values in `# Sub-agent Budget` are load-bearing claims. If the smoke test reveals that the launcher exported `MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS=15` or `MILES_PARL_V2_SUBAGENT_ACCESS_CHARS=8000` (or similar non-default), the prompt text must be updated to match. Do NOT update the launcher to match the prompt — the launcher is the source of truth, the prompt follows. Also verify `SUBAGENT_REACT_SUFFIX` in `widesearch/subagent_prompts.py:38` still states the same 10-call cap it auto-injects into sub-agents; that number must match the one we quote at Orchestrator level.

- [ ] **Step 4: Repeat for the other two launchers (lightweight)**

```bash
MODE=debug_minimal bash examples/parl_v2/widesearch/run-qwen3-4B-widesearch.sh
MODE=debug_minimal bash examples/parl_v2/widesearch/run-qwen3-4B-widesearch-single.sh
```

Confirm each logs its respective rewritten prompt, and none of the prompts mentions tools that mode does not actually expose (e.g. swarm-strict log must NOT contain any `search`/`access` tool-spec block in its injected tools).

- [ ] **Step 5: No commit needed if Step 3 didn't trigger a fix**

If Step 3 required a prompt correction, commit it as:

```bash
git add examples/parl_v2/prompts.py
git commit -m "[parl_v2/prompts] sync budget numbers with actual launcher env"
```

---

## Self-Review (plan-author sanity check)

**Spec coverage:** every auditable issue from the prompt re-evaluation maps to a task —

- Factual error "refines"→replaces, vague "key findings"→`<result>` → Task 2.
- Missing disclosure: no-direct-tools statement → Task 2.
- Missing disclosure: sub-agent budget (10 tool calls / 5000 chars) → Tasks 1 and 2.
- Missing disclosure: 8-name cap reuse semantics → Tasks 1 and 2.
- Adversarial heuristic "call directly when 1-2 lookups" → Task 1.
- Adversarial heuristic "Budget conservatively" (single-agent step 5) → Task 3.
- Workflow bug: single-agent step 4 loops to wrong step → Task 3.
- Style: role string harmonization → Tasks 1/2/3 in lockstep.
- Ruled-out: `# Sub-agent Validity` section → explicitly NOT added in Tasks 1/2 (redundant with auto-injected sub-agent suffix). Task 4 docstring calls this out so the next editor does not re-add it.
- End-to-end confirmation that rewrites actually reach the model and budget numbers match runtime → Task 5.

**Placeholder scan:** no "TBD"/"fill in later"/"handle edge cases"; every prompt body is the full literal string an engineer copies; every commit message is the full literal text.

**Consistency:**
- Role string "expert in information collection and organization" identical across Tasks 1, 2, 3.
- Sub-agent budget numbers (10 tool calls total, ~5000 chars per access) identical between Tasks 1 and 2; Task 3 has no sub-agents.
- `<result>…</result>` contract phrasing identical in `# Context Isolation` of Tasks 1 and 2.
- Neither Task 1 nor Task 2 restates the sub-agent-side `<result>` / must-tool-use contract (it is auto-injected via `SUBAGENT_OUTPUT_SUFFIX` + `SUBAGENT_REACT_SUFFIX`).
- "ReAct loop" is the shared framing in Tasks 1 and 2 for how sub-agents behave internally; neither spells out the workflow, by design.

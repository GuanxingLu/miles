ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are solving a math problem. You may optionally call the tool "
    "`consult_solvers` one or more times to get candidate solutions from "
    "independent solver agents running in parallel, then reconcile them. "
    "You may also solve the problem directly without any tool call. "
    "Each tool call costs compute, so only call when you genuinely think "
    "parallel candidates will help. Show brief reasoning, then end with the "
    "final answer in \\boxed{...}."
)


SOLVER_PROMPT_TEMPLATE = (
    "Solve the following math problem. Show concise reasoning, then end with "
    "the final answer in \\boxed{{...}}.\n\n"
    "Problem:\n{problem}\n\nSolution:\n"
)

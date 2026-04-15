SOLVER_PROMPT_TEMPLATE = """{problem_statement}"""


def generate_orchestrator_template(num_solutions: int) -> str:
    """Orchestrator prompt: aggregates N candidate solutions into a final answer."""
    sections = []
    for i in range(num_solutions):
        sections.append(f"#### Candidate Solution {i+1}\n{{solution{i+1}}}\n\n---")
    solutions_text = "\n".join(sections)

    return f"""You are given a math problem and {num_solutions} candidate solutions produced independently by other solvers. Some may be wrong or inconsistent.

Your task: carefully analyze the candidates, reconcile disagreements, and produce the correct final answer. You may rely on any candidate's reasoning, combine insights, or reason from scratch. Show brief reasoning, then give the final answer in \\boxed{{{{...}}}}.

### Problem

{{problem_statement}}

---

### Candidate Solutions
{solutions_text}

---

Now produce your final, correct solution. End with the answer in \\boxed{{{{...}}}}.
"""

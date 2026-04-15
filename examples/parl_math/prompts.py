SOLVER_PROMPT_TEMPLATE = """{problem_statement}"""


def _orchestrator_user_content(problem_statement: str, candidate_texts: list[str]) -> str:
    sections = []
    for i, text in enumerate(candidate_texts):
        sections.append(f"#### Candidate Solution {i+1}\n{text}\n\n---")
    solutions_text = "\n".join(sections)
    num_solutions = len(candidate_texts)

    return f"""You are given a math problem and {num_solutions} candidate solutions produced independently by other solvers. Some may be wrong or inconsistent.

Your task: carefully analyze the candidates, reconcile disagreements, and produce the correct final answer. You may rely on any candidate's reasoning, combine insights, or reason from scratch. Show brief reasoning, then give the final answer in \\boxed{{...}}.

### Problem

{problem_statement}

---

### Candidate Solutions
{solutions_text}

---

Now produce your final, correct solution. End with the answer in \\boxed{{...}}."""


def build_orchestrator_prompt(tokenizer, problem_statement: str, candidate_texts: list[str]) -> str:
    user_content = _orchestrator_user_content(problem_statement, candidate_texts)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

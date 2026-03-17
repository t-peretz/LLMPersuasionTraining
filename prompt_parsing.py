"""Prompt builders and parsing helpers for agent A/B IO formats."""

import re
from typing import Dict, List, Optional


def build_A_prompt(question: str) -> List[Dict[str, str]]:
    system = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
            The assistant first thinks about the reasoning process and then provides the answer.
            Your final answer should be a single number.
            The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags.

            Output format must be exactly:
            <think> reasoning process here </think>
            <answer> single numerical answer here </answer>"""
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def build_B_prompt(question: str, a_output: str) -> List[Dict[str, str]]:
    system = """A conversation between User and Assistant. The user provides:
          1) A question.
          2) An argument that attempts to answer the question, including reasoning and a final answer.

          The Assistant evaluates whether the argument correctly and logically answers the given question.

          The Assistant must first analyze the argument with respect to the question.
          This analysis must be enclosed within <analysis> </analysis> tags.
          Then the Assistant must give a final judgment.

          Valid judgments are only:
          RIGHT
          WRONG

          Very important:
          - Do not include anything outside the required tags.
          - The <judgement> tag must contain exactly one word: RIGHT or WRONG.
          - The output format must be exactly:

          <analysis> evaluation of whether the argument correctly answers the question </analysis>
          <judgement> RIGHT or WRONG </judgement>"""
    user = f"Question:\n{question}\n\nAgent A output:\n{a_output}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL)
ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL)
FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", flags=re.DOTALL)
B_JUDGEMENT_RE = re.compile(r"<judge?ment>\s*(.*?)\s*</judge?ment>", flags=re.DOTALL | re.IGNORECASE)
B_FORMAT_RE = re.compile(r"<analysis>.*?</analysis>\s*<judge?ment>.*?</judge?ment>", flags=re.DOTALL | re.IGNORECASE)
ANALYSIS_RE = re.compile(r"<analysis>\s*(.*?)\s*</analysis>", flags=re.DOTALL)
STRICT_NUM_RE = re.compile(r"^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*$")


def extract_think(text: str) -> Optional[str]:
    m = THINK_RE.search(text)
    return m.group(1).strip() if m else None


def extract_answer_number(text: str) -> Optional[float]:
    m = ANS_RE.search(text)
    if not m:
        return None
    ans = m.group(1).strip()
    if not STRICT_NUM_RE.match(ans):
        return None
    return float(ans)


def is_correct_num(pred: Optional[float], gold: str, tol: float = 1e-9) -> bool:
    if pred is None:
        return False
    try:
        g = float(gold)
    except ValueError:
        return False
    return abs(pred - g) <= tol


def has_think_answer_format(text: str) -> bool:
    s = text.strip()
    if FORMAT_RE.fullmatch(s) is None:
        return False
    return True


def has_analysis_judgement_format(text: str) -> bool:
    s = text.strip()
    if B_FORMAT_RE.fullmatch(s) is None:
        return False
    return True


def parse_B_label(text: str) -> Optional[str]:
    m = B_JUDGEMENT_RE.search(text)
    if not m:
        return None
    lab = m.group(1).strip().upper()
    if lab in ("RIGHT", "WRONG"):
        return lab
    return None

"""Evaluate trained Agent A on the GSM8K test set, optionally with Agent B judging."""

import argparse
import json
import re

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from prompt_parsing import (
    build_A_prompt,
    build_B_prompt,
    extract_answer_number,
    is_correct_num,
    parse_B_label,
)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def test_model(
    checkpoint_a: str,
    checkpoint_b: str = None,
    num_questions: int = None,
    output_path: str = "test_results.json",
) -> dict:
    # Load GSM8K test set
    ds = load_dataset("openai/gsm8k", "main")
    test = ds["test"]

    questions = []
    golds = []
    for ex in test:
        m = re.search(r"####\s*([-\d\.]+)", ex["answer"])
        if not m:
            continue
        questions.append(ex["question"])
        golds.append(m.group(1))

    if num_questions is not None:
        questions = questions[:num_questions]
        golds = golds[:num_questions]

    total = len(golds)

    print("Loading base model with LoRA serving...")
    llm = LLM(
        model=MODEL_ID,
        enable_lora=True,
        max_lora_rank=8,
        dtype="auto",
    )
    tok = llm.get_tokenizer()
    lora_a = LoRARequest("agent_a", 1, checkpoint_a)

    # --- Agent A ---
    prompts_a = [
        tok.apply_chat_template(build_A_prompt(q), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    print(f"Running Agent A inference on {total} examples...")
    outputs_a = llm.generate(prompts_a, SamplingParams(max_tokens=512, temperature=0), lora_request=lora_a)
    a_texts = [o.outputs[0].text for o in outputs_a]

    a_correct = 0
    records = []
    for i, text in enumerate(a_texts):
        a_accuracy_reward = int(is_correct_num(extract_answer_number(text), golds[i]))
        a_correct += a_accuracy_reward
        records.append({
            "question": questions[i],
            "ground_truth": golds[i],
            "a_output": text,
            "a_accuracy_reward": a_accuracy_reward,
        })

    a_accuracy = a_correct / total
    print(f"\nAgent A accuracy: {a_correct}/{total} = {a_accuracy:.4f}")

    stats = {"a_accuracy": a_accuracy}

    if checkpoint_b is None:
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Results saved to {output_path}")
        return stats

    # --- Agent B ---
    lora_b = LoRARequest("agent_b", 2, checkpoint_b)
    prompts_b = [
        tok.apply_chat_template(
            build_B_prompt(questions[i], a_texts[i]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(total)
    ]
    print(f"Running Agent B inference on {total} examples...")
    outputs_b = llm.generate(prompts_b, SamplingParams(max_tokens=256, temperature=0), lora_request=lora_b)
    b_texts = [o.outputs[0].text for o in outputs_b]

    b_correct = 0
    b_positive = 0
    b_negative = 0
    for i in range(total):
        a_right = bool(records[i]["a_accuracy_reward"])
        b_label = parse_B_label(b_texts[i])
        gold_label = "RIGHT" if a_right else "WRONG"
        b_reward = int(b_label == gold_label)
        b_correct += b_reward
        if b_label == "RIGHT":
            b_positive += 1
        else:
            b_negative += 1
        records[i].update({
            "b_output": b_texts[i],
            "b_score": b_label,
            "b_reward": b_reward,
        })

    b_accuracy = b_correct / total
    print(f"Agent B accuracy: {b_correct}/{total} = {b_accuracy:.4f}")
    print(f"Agent B positive (said RIGHT): {b_positive}/{total}")
    print(f"Agent B negative (said WRONG): {b_negative}/{total}")

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Results saved to {output_path}")

    stats.update({
        "b_accuracy": b_accuracy,
        "b_positive": b_positive,
        "b_negative": b_negative,
    })
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K test set")
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--checkpoint_b", type=str, default=None, help="Agent B checkpoint (optional)")
    parser.add_argument("--num_questions", type=int, default=None, help="Only test the first N questions (default: all)")
    parser.add_argument("--output", type=str, default="test_results.json", help="Path to save per-question results JSON")
    args = parser.parse_args()

    test_model(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        num_questions=args.num_questions,
        output_path=args.output,
    )

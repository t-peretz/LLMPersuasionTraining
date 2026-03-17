"""Alternating trainer: Agent A (answering the question) and Agent B (judging the answer)."""

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import torch
from trl import GRPOTrainer
from transformers import TrainerCallback

from config import Configuration
from prompt_parsing import (
    build_B_prompt,
    extract_answer_number,
    extract_think,
    has_analysis_judgement_format,
    has_think_answer_format,
    is_correct_num,
    parse_B_label,
)


class ReinforceBCallback(TrainerCallback):
    """
    Runs B reinforcement every k optimizer steps of A, and handles all logging,
    checkpointing, and metrics saving.

    Training parameters (log_every, verbose, etc.) are set by
    AlternatingGRPOJudgeTrainer.train() before calling trainer_A.train().
    """

    def __init__(self, owner, k: int):
        self.owner = owner
        self.k = int(k)
        # Configured by train() before trainer_A.train() starts
        self.log_every: int = 1
        self.verbose: bool = False
        self.save_every: int = 5
        self.save_dir: str = "checkpoints"
        self.num_alternations: int = 0
        self.metrics_path: str = "training_metrics.json"
        # Runtime state
        self.history: List[Dict[str, Any]] = []
        self.b_updates_count: int = 0
        self._start_time: Optional[float] = None
        self._window_start: Optional[float] = None
        self._last_logs: Dict[str, Any] = {}
        self._last_A_metrics: Dict[str, Any] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        self.b_updates_count = 0
        self.history = []
        self._start_time = self._window_start = time.time()
        print("\n" + "=" * 80)
        print(f"Starting Alternating Training: {self.num_alternations} B updates | A steps per B: {self.k}")
        print("=" * 80 + "\n")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self._last_logs = dict(logs)
            # Cache A metrics here — on_log fires after on_step_end, so this is
            # the earliest point where the current step's metrics are available.
            # Try both key formats TRL may emit (with/without /mean, with/without train/ prefix).
            self._last_A_metrics = {
                "global_step":     state.global_step,
                "loss":            logs.get("loss") or logs.get("train/loss"),
                "reward":          logs.get("reward") or logs.get("train/reward"),
                "reward_std":      logs.get("reward_std") or logs.get("train/reward_std"),
                "format_reward":   logs.get("rewards/format_reward_think_answer") or logs.get("rewards/format_reward_think_answer/mean"),
                "accuracy_reward": logs.get("rewards/accuracy_reward_A") or logs.get("rewards/accuracy_reward_A/mean"),
                "judge_reward":    logs.get("rewards/judge_by_B") or logs.get("rewards/judge_by_B/mean"),
            }
        return control

    def extract_A_metrics(self, state) -> Dict[str, Any]:
        # Return metrics cached by on_log from the previous step.
        # on_step_end always fires before on_log for the same step, so the most
        # recent available A metrics are from the prior step.
        return self._last_A_metrics or {"global_step": state.global_step}

    def print_step(self, local_b_step: int, global_step: int, metrics_A: Dict, metrics_B: Dict):
        elapsed = time.time() - self._start_time
        step_time = time.time() - self._window_start
        print("\n" + "─" * 80)
        print(f"Step {local_b_step}/{self.num_alternations} (A global_step={global_step}) | "
              f"Elapsed: {elapsed:.1f}s | Step time: {step_time:.1f}s")
        print("─" * 80)
        print("Agent A (GRPO):")
        print(f"  Loss:          {metrics_A.get('loss', 'N/A')}")
        print(f"  Total Reward:  {metrics_A.get('reward', 'N/A')}")
        print(f"  Reward Std:    {metrics_A.get('reward_std', 'N/A')}")
        print(f"  Format Reward: {metrics_A.get('format_reward', 'N/A')}")
        print(f"  Accuracy Rew:  {metrics_A.get('accuracy_reward', 'N/A')}")
        print(f"  Judge Reward:  {metrics_A.get('judge_reward', 'N/A')}")
        print("\nAgent B (Reinforcement):")
        print(f"  Weighted Loss: {metrics_B['loss']:.4f}")
        print(f"  Examples:      {metrics_B['num_examples']}")
        print(f"  Avg |weight|:  {metrics_B['avg_weight']:.3f}")
        print(f"  Judge Acc:     {metrics_B['judge_accuracy']:.3f}")
        print(f"  Format Reward: {metrics_B['format_reward']:.3f}")
        print("─" * 80 + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        if self.k <= 0 or state.global_step <= 0 or state.global_step % self.k != 0:
            return control

        self.owner.set_train_phase("B")
        try:
            metrics_B = self.owner.reinforce_B()
        finally:
            self.owner.set_train_phase("A")
            torch.cuda.empty_cache()

        self.b_updates_count += 1
        local_b_step = self.b_updates_count

        if self.verbose:
            self.owner.print_verbose_step_example(state.global_step)
        self.owner.cached_B_data = []
        self.owner.step_debug_examples = []

        metrics_A = self.extract_A_metrics(state)

        if local_b_step % self.log_every == 0:
            self.print_step(local_b_step, state.global_step, metrics_A, metrics_B)

        if self.save_every > 0 and local_b_step > 0 and local_b_step % self.save_every == 0:
            self.owner.save_models(os.path.join(self.save_dir, f"step_{local_b_step}"))

        self.history.append({
            "step": local_b_step,
            "agent_a_global_step":      state.global_step,
            "elapsed_sec":              float(time.time() - self._start_time),
            "step_time_sec":            float(time.time() - self._window_start),
            "agent_a_loss":             metrics_A.get("loss"),
            "agent_a_reward":           metrics_A.get("reward"),
            "agent_a_reward_std":       metrics_A.get("reward_std"),
            "agent_a_format_reward":    metrics_A.get("format_reward"),
            "agent_a_accuracy_reward":  metrics_A.get("accuracy_reward"),
            "agent_a_judge_reward":     metrics_A.get("judge_reward"),
            "agent_b_weighted_loss":    metrics_B["loss"],
            "agent_b_examples":         metrics_B["num_examples"],
            "agent_b_avg_abs_weight":   metrics_B["avg_weight"],
            "agent_b_judge_accuracy":   metrics_B["judge_accuracy"],
            "agent_b_format_reward":    metrics_B["format_reward"],
        })
        self._window_start = time.time()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        print("\n" + "=" * 80)
        print(f"Training Complete! Total time: {time.time() - self._start_time:.1f}s")
        print("=" * 80 + "\n")
        if self.metrics_path and self.history:
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            print(f"saved metrics: {self.metrics_path}")
        return control


class AlternatingGRPOJudgeTrainer:
    """
    Alternating training:
      - A: GRPOTrainer with three rewards (format, accuracy, B-judge) + B output caching
      - B: GRPO style training on its own evaluation outputs

    cached_B_data item schema:
      { "full_ids": LongTensor,  "old_logps": FloatTensor,  "prefix_width": int,
        "weight": float,  "b_correct": bool,  "format_ok": bool,  "question_key": str }
    """

    def __init__(
        self,
        model_A,
        model_B,
        tokenizer,
        dataset_A,
        config_A,
        configuration: Optional[Configuration] = None,
    ):
        self.model_A = model_A
        self.model_B = model_B
        self.tokenizer = tokenizer
        self.cfg = configuration or Configuration()

        self.b_batch_size = int(self.cfg.B_BATCH_SIZE)
        self.max_b_train_tokens    = int(self.cfg.MAX_B_TRAIN_TOKENS)
        self.correct_weight        = float(self.cfg.B_CORRECT_REWARD)
        self.format_weight         = float(self.cfg.FORMAT_REWARD)
        self.grad_clip_norm        = float(self.cfg.GRAD_CLIP_NORM)
        self.temperature_B         = float(self.cfg.B_TEMPERATURE)
        self.top_p_B               = float(self.cfg.B_TOP_P)
        self.clip_eps              = float(self.cfg.B_CLIP_EPS)

        self.cached_B_data:      List[Dict[str, Any]] = []
        self.step_debug_examples: List[Dict[str, Any]] = []

        # Optimizer captures only B's LoRA params (requires_grad=True at this point)
        self.optimizer_B = torch.optim.AdamW(
            (p for p in self.model_B.parameters() if p.requires_grad),
            lr=float(self.cfg.B_LEARNING_RATE),
            weight_decay=0.0,
        )

        fmt_reward = self.cfg.FORMAT_REWARD
        acc_reward = self.cfg.A_ACCURACY_REWARD

        def format_reward_think_answer(completions, **kwargs):
            texts = [c[0]["content"] for c in completions]
            return [fmt_reward if has_think_answer_format(t) else 0.0 for t in texts]

        def accuracy_reward_A(completions, solution, **kwargs):
            texts = [c[0]["content"] for c in completions]
            return [
                acc_reward if is_correct_num(extract_answer_number(t), sol) else 0.0
                for t, sol in zip(texts, solution)
            ]

        def judge_by_B(completions, question, solution, **kwargs):
            return self.evaluate_with_B(completions, question, solution)

        self.trainer_A = GRPOTrainer(
            model=self.model_A,
            args=config_A,
            train_dataset=dataset_A,
            processing_class=self.tokenizer,
            reward_funcs=[format_reward_think_answer, accuracy_reward_A, judge_by_B],
        )
        self.b_callback = ReinforceBCallback(self, int(self.cfg.A_STEPS_PER_B))
        self.trainer_A.add_callback(self.b_callback)
        self.set_train_phase("A")

    def set_train_phase(self, active: str) -> None:
        """Freeze the inactive model; unfreeze only LoRA params in the active one."""
        freeze, thaw = (self.model_B, self.model_A) if active == "A" else (self.model_A, self.model_B)
        freeze.requires_grad_(False)
        for name, p in thaw.named_parameters():
            p.requires_grad = "lora_" in name

    def print_verbose_step_example(self, step_num: int) -> None:
        if not self.step_debug_examples:
            print(f"[verbose] Step {step_num}: no valid A/B debug example captured.")
            return
        ex = self.step_debug_examples[0]
        print("\n[verbose] Example from completed step")
        print(f"  Step:           {step_num}")
        print(f"  correct label:  {ex.get('correct_label')}")
        print(f"  B label:        {ex.get('b_label')}")
        print(f"  B format ok?:   {ex.get('format_ok')}")
        print(f"  B reward:       {ex.get('b_reward')}")
        print(f"  B weight:       {ex.get('b_weight')}")
        print(f"  A judge reward: {ex.get('a_judge_reward')}")
        print(f"  A answer:       {ex.get('a_answer')}")
        print(f"  Solution:       {ex.get('solution')}")
        print(f"  Question:       {ex.get('question')}")
        print(f"  A output:       {ex.get('a_output')}")
        print(f"  B output:       {ex.get('b_output')}")

    @torch.no_grad()
    def evaluate_with_B(self, completions, question, solution):
        texts = [c[0]["content"] for c in completions]
        rewards = [0.0] * len(texts)

        # Build B prompts only for A outputs that have valid format and a think section.
        batch_prompts, meta = [], []  # meta: (orig_i, correct_label, question_key)
        for i, (q, sol, a_text) in enumerate(zip(question, solution, texts)):
            if not has_think_answer_format(a_text):
                continue
            a_think = extract_think(a_text)
            if not a_think:
                continue
            correct_label = "RIGHT" if is_correct_num(extract_answer_number(a_text), sol) else "WRONG"
            batch_prompts.append(build_B_prompt(q, a_text))
            meta.append((i, correct_label, str(q)))

        if not batch_prompts:
            return rewards

        self.model_B.eval()
        rendered = self.tokenizer.apply_chat_template(batch_prompts, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(rendered, return_tensors="pt", padding=True)
        enc = {k: v.to(self.model_B.device) for k, v in enc.items()}

        if self.tokenizer.padding_side != "left":
            raise RuntimeError(f"B judge batching requires left padding; got {self.tokenizer.padding_side!r}")

        # With left padding, all prompts end at the same column, making prefix_width a
        # shared label-mask boundary for every row in this batch.
        prefix_width = enc["input_ids"].shape[1]
        out = self.model_B.generate(
            **enc,
            max_new_tokens=int(self.cfg.B_MAX_TOKENS),
            do_sample=True,
            temperature=self.temperature_B,
            top_p=self.top_p_B,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, prefix_width:]

        # Compute old (behavior-policy) per-token log-probs for the clipped PG update.
        fwd = self.model_B(input_ids=out, attention_mask=(out != self.tokenizer.pad_token_id).long())
        logits = fwd.logits[:, :-1]  # (B, T-1, V)
        target = out[:, 1:]          # (B, T-1)
        old_logps = torch.log_softmax(logits, dim=-1)
        old_per_token_logps = old_logps.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # Pass 1: decode B outputs, compute per-example rewards, accumulate per-question rewards.
        scored = []
        rewards_by_q: Dict[str, List[float]] = defaultdict(list)
        for b_idx, (orig_i, correct_label, question_key) in enumerate(meta):
            b_output = self.tokenizer.decode(gen[b_idx], skip_special_tokens=True)
            b_label  = parse_B_label(b_output)
            fmt_ok   = has_analysis_judgement_format(b_output)
            b_correct = b_label == correct_label
            b_reward  = (self.correct_weight if b_correct else 0.0) + (self.format_weight if fmt_ok else 0.0)

            rewards[orig_i] = float(self.cfg.A_PERSUASION_REWARD) if b_label == "RIGHT" else 0.0
            rewards_by_q[question_key].append(b_reward)
            scored.append({
                "orig_i":       orig_i,
                "full_ids":     out[b_idx].detach().to("cpu"),
                "old_logps":    old_per_token_logps[b_idx].detach().to("cpu"),
                "prefix_width": int(prefix_width),
                "reward":       b_reward,
                "question_key": question_key,
                "b_correct":    b_correct,
                "format_ok":    fmt_ok,
                "correct_label": correct_label,
                "b_output":     b_output,
                "b_label":      b_label,
                "a_judge_reward": float(rewards[orig_i]),
            })

        # Pass 2: compute per-question-centered weights and populate caches.
        mean_reward = {q: sum(rs) / len(rs) for q, rs in rewards_by_q.items()}
        for item in scored:
            q = item["question_key"]
            w = item["reward"] - mean_reward[q]
            if not self.step_debug_examples:
                orig_i = item["orig_i"]
                self.step_debug_examples.append({
                    "question":      question[orig_i],
                    "solution":      solution[orig_i],
                    "a_output":      texts[orig_i],
                    "a_answer":      extract_answer_number(texts[orig_i]),
                    "correct_label": item["correct_label"],
                    "b_output":      item["b_output"],
                    "b_label":       item["b_label"],
                    "format_ok":     item["format_ok"],
                    "b_reward":      item["reward"],
                    "b_weight":      w,
                    "a_judge_reward":item["a_judge_reward"],
                })
            self.cached_B_data.append({
                "full_ids":     item["full_ids"],
                "old_logps":    item["old_logps"],
                "prefix_width": item["prefix_width"],
                "weight":       float(w),
                "b_correct":    item["b_correct"],
                "format_ok":    item["format_ok"],
                "question_key": q,
            })

        return rewards

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_id: int) -> torch.Tensor:
        """Right-pad 1D LongTensors to the same length."""
        max_len = max(s.numel() for s in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            out[i, :seq.numel()] = seq
        return out

    def reinforce_B(self) -> Dict[str, Any]:
        if not self.cached_B_data:
            return {"loss": 0.0, "num_examples": 0, "avg_weight": 0.0,
                    "judge_accuracy": 0.0, "format_reward": 0.0}

        self.model_B.train()

        # Normalize: weight each example by 1/k_q (where k_q = completions for question q).
        q_counts = Counter(b["question_key"] for b in self.cached_B_data)
        num_questions = max(len(q_counts), 1)
        q_norm = {q: max(cnt, 1) for q, cnt in q_counts.items()}

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad_token_id or eos_token_id")

        total_obj = 0.0

        for i in range(0, len(self.cached_B_data), self.b_batch_size):
            batch = self.cached_B_data[i : i + self.b_batch_size]
            weights = torch.tensor(
                [b["weight"] / q_norm[b["question_key"]] for b in batch],
                device=self.model_B.device, dtype=torch.float32,
            )
            if weights.abs().max() < 1e-6:
                continue

            # Truncate from the left to cap memory; preserves the generation tail.
            seqs, old_logps_list, prefix_widths = [], [], []
            for b in batch:
                seq, pw = b["full_ids"], int(b["prefix_width"])
                old_lp = b["old_logps"]  # (T-1,)
                if self.max_b_train_tokens > 0 and seq.numel() > self.max_b_train_tokens:
                    cut = seq.numel() - self.max_b_train_tokens
                    seq, pw = seq[cut:], max(0, pw - cut)
                    # old_logps has length T-1, aligns with positions 0..T-2 predicting tokens 1..T-1
                    old_lp = old_lp[cut:]
                seqs.append(seq)
                old_logps_list.append(old_lp)
                prefix_widths.append(pw)

            input_ids = self.pad_sequences(seqs, pad_id).to(self.model_B.device)
            attention_mask = (input_ids != pad_id)

            # Build completion mask: True only for generated tokens (after prefix, not padding)
            completion_mask = torch.zeros_like(attention_mask)
            for j, pw in enumerate(prefix_widths):
                completion_mask[j, pw:] = True
            completion_mask = completion_mask & attention_mask
            # Shift to align with logits[:, :-1] predicting input_ids[:, 1:]
            shift_completion_mask = completion_mask[:, 1:].contiguous()

            # Pad old log-probs to match input_ids length (T-1 positions)
            max_len = input_ids.shape[1] - 1
            old_logps_padded = torch.zeros(len(batch), max_len, device=self.model_B.device)
            for j, old_lp in enumerate(old_logps_list):
                # old_lp aligns with the non-padded portion; right-pad has zeros (masked out)
                L = min(old_lp.numel(), max_len)
                old_logps_padded[j, :L] = old_lp[:L].to(self.model_B.device)

            # Forward pass: compute current per-token log-probs
            logits = self.model_B(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = input_ids[:, 1:].contiguous()
            new_logps = torch.log_softmax(shift_logits, dim=-1)
            new_per_token_logps = new_logps.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)

            # Clipped policy gradient
            log_ratio = new_per_token_logps - old_logps_padded
            ratio = torch.exp(log_ratio)
            eps = self.clip_eps
            clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)

            # advantages shape (batch,) -> (batch, 1) for broadcasting
            advantages = weights.unsqueeze(1)
            per_token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

            # Mean over completion tokens per example, then weight-sum over examples
            per_ex_loss = (per_token_loss * shift_completion_mask).sum(dim=1) / shift_completion_mask.sum(dim=1).clamp_min(1)
            obj = per_ex_loss.sum()
            batch_q_count = max(len({b["question_key"] for b in batch}), 1)
            self.optimizer_B.zero_grad(set_to_none=True)
            (obj / batch_q_count).backward()
            total_obj += float(obj.detach())
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model_B.parameters(), self.grad_clip_norm)
            self.optimizer_B.step()

        n = len(self.cached_B_data)
        return {
            "loss":          total_obj / num_questions,
            "num_examples":  n,
            "avg_weight":    sum(abs(b["weight"]) for b in self.cached_B_data) / max(n, 1),
            "judge_accuracy":sum(b["b_correct"] for b in self.cached_B_data) / max(n, 1),
            "format_reward": sum(self.format_weight * b["format_ok"] for b in self.cached_B_data) / max(n, 1),
        }

    def train(
        self,
        num_alternations: int,
        log_every: int = 1,
        verbose: bool = False,
        save_every: int = 5,
    ):
        if int(self.cfg.A_STEPS_PER_B) <= 0:
            raise ValueError("A_STEPS_PER_B must be > 0")

        self.b_callback.log_every        = log_every
        self.b_callback.verbose          = verbose
        self.b_callback.save_every       = save_every
        self.b_callback.save_dir         = "checkpoints"
        self.b_callback.num_alternations = num_alternations
        self.b_callback.metrics_path     = "training_metrics.json"

        self.trainer_A.args.max_steps = num_alternations * int(self.cfg.A_STEPS_PER_B)
        self.cached_B_data = []
        self.step_debug_examples = []
        self.set_train_phase("A")

        self.trainer_A.train()

    def save_models(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model_A.save_pretrained(os.path.join(output_dir, "agent_A"))
        self.model_B.save_pretrained(os.path.join(output_dir, "agent_B"))
        print(f"saved: {output_dir}/agent_A and {output_dir}/agent_B")

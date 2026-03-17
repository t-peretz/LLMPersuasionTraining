import logging
import random
import warnings
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from config import Configuration
from dataset import ListDataset, load_gsm8k
from model import AlternatingGRPOJudgeTrainer


def load_base_model(model_id: str, name: str):
    last_err = None
    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation=attn_impl,
            )
            print(f"{name}: loaded with attn_implementation={attn_impl}")
            return model
        except Exception as e:
            last_err = e
            print(f"{name}: attn_implementation={attn_impl} failed ({e}); trying fallback.")
    if last_err:
        print(f"{name}: falling back to default attention implementation.")
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )


def training_pipeline(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    num_steps: int = 200,
    seed: int = 0,
    a_lr: float = 1e-5,
    configuration: Optional[Configuration] = None,
    verbose: bool = True,
):
    cfg = configuration or Configuration()

    random.seed(seed)
    torch.manual_seed(seed)

    # Load GSM8K
    rows_A = load_gsm8k()
    dataset_A = ListDataset(rows_A)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # A/GRPO can stay left-padded; B finetune uses cached ids (we right-pad ourselves)
    tok.padding_side = "left"

    base_A = load_base_model(model_id, "base_A")
    base_B = load_base_model(model_id, "base_B")

    # LoRA
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model_A = get_peft_model(base_A, lora)
    model_B = get_peft_model(base_B, lora)

    try:
        model_A.print_trainable_parameters()
        model_B.print_trainable_parameters()
    except Exception:
        pass

    # Suppress excessive warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # GRPO config for A only
    config_A = GRPOConfig(
        output_dir="A_out",
        per_device_train_batch_size=int(cfg.A_BATCH_SIZE),
        gradient_accumulation_steps=1,
        learning_rate=a_lr,
        bf16=True,
        remove_unused_columns=False,
        num_generations=int(cfg.A_NUM_GENERATIONS),
        max_completion_length=int(cfg.A_MAX_TOKENS),
        temperature=float(cfg.A_TEMPERATURE),
        top_p=float(cfg.A_TOP_P),
        logging_steps=int(cfg.A_STEPS_PER_B),
        report_to=[],
        logging_strategy="steps",
        use_vllm=True,
        vllm_mode="colocate",
    )

    trainer = AlternatingGRPOJudgeTrainer(
        model_A=model_A,
        model_B=model_B,
        tokenizer=tok,
        dataset_A=dataset_A,
        config_A=config_A,
        configuration=cfg,
    )

    trainer.train(
        num_alternations=num_steps,
        log_every=1,
        verbose=bool(verbose),
    )

    return None

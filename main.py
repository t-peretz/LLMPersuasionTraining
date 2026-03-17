import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from train import training_pipeline

if __name__ == "__main__":
    training_pipeline(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        num_steps=200,
    )

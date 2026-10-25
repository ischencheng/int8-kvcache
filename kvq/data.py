"""Datasets and chat prompts shared by capture, calibration and evaluation."""
import random

from datasets import load_dataset

BOXED = "Please reason step by step, and put your final answer within \\boxed{}."


def aime24():
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    return [{"id": f"aime24-{r['id']}", "question": r["problem"], "answer": str(r["answer"])} for r in ds]


def gsm8k(n=None, seed=0):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = [{"id": f"gsm8k-{i}", "question": r["question"],
             "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(ds)]
    if n is not None:
        rows = random.Random(seed).sample(rows, n)
    return rows


def math_messages(question):
    return [{"role": "user", "content": f"{question}\n{BOXED}"}]

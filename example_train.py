from transformers import (
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from peft.utils.config import TaskType

from model_util import load_stablelm_models, create_peft_model
from dataset_util import create_dataset

PRETRAINED_MODEL = "stabilityai/japanese-stablelm-base-alpha-7b"
DATASET_NAME = "YOUR_DATASET_NAME"
VAL_SPLIT_SIZE = 0.2  # 検証データの割合

PEFT_NAME = "my_stable_peft"  # save name
OUTPUT_DIR = f"./{PEFT_NAME}_output"

LORA_CONFIG = {
    "r": 8,  # LoRA の rank
    "lora_alpha": 1,
    "lora_dropout": 0.01,
    "inference_mode": False,
    "task_type": TaskType.CAUSAL_LM,
    "target_modules": ["query_key_value"],
}

CUTOFF_LENGTH = 512  # コンテキスト上限

# 学習パラメータ
EVAL_STEPS = 50  # 検証の間隔
SAVE_STEPS = 100  # モデルの保存間隔
MAX_STEPS = 300  # 学習の最大ステップ数
REPORT_TO = "wandb"


def preprocess_prompt(prompt: str):
    return f"{prompt}<|endoftext|>"


def tokenize(prompt: str, tokenizer: PreTrainedTokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LENGTH,
        padding=False,
    )

    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }


def compose_prompt(
    data_point: dict[str, any], tokenizer: PreTrainedTokenizer
) -> tuple[str, str]:
    ## FIXME* Customize here
    instruction = data_point["instruction"]
    output = data_point["output"]
    prompts = [
        "### instruction:",
        instruction,
        "\n\n",
        "### output:",
        output,
    ]
    prompt = tokenize(preprocess_prompt("\n".join(prompts)), tokenizer)

    return prompt


def main():
    tokenizer, model = load_stablelm_models(PRETRAINED_MODEL)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model = create_peft_model(model, LoraConfig(**LORA_CONFIG))  # peft 適用

    tran_data, val_data = create_dataset(
        DATASET_NAME,
        map_action=lambda x: compose_prompt(x, tokenizer),
        val_split=VAL_SPLIT_SIZE,
    )

    training_args = TrainingArguments(
        num_train_epochs=3,
        learning_rate=3e-4,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        report_to=REPORT_TO,
        push_to_hub=False,
        auto_find_batch_size=True,
    )
    trainer = Trainer(
        model=model,
        train_dataset=tran_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 学習
    trainer.train()

    # モデルの保存
    trainer.model.save_pretrained(PEFT_NAME)
    tokenizer.save_pretrained(PEFT_NAME)

    print("Done!")


if __name__ == "__main__":
    main()

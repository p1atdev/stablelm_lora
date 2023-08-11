import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (
    LlamaTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model

NERDSTASH_TOKENIZER_V1 = "novelai/nerdstash-tokenizer-v1"


def load_stablelm_models(
    pretrained_model_name_or_path: str,
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer = LlamaTokenizer.from_pretrained(
        NERDSTASH_TOKENIZER_V1,
        add_special_tokens=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True,
    )

    return tokenizer, model


def create_peft_model(model: PreTrainedModel, config: LoraConfig):
    return get_peft_model(
        model,
        config,
    )

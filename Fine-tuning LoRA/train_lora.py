#!/usr/bin/env python3
"""
Entrenamiento LoRA para crear un tutor de programación.
Funciona 100% local con GPU Nvidia (8-12 GB VRAM).
Sin GPU, funciona en CPU pero será mucho más lento.
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar LoRA para tutor de programación")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Modelo base de HuggingFace (default: Llama 3.2 3B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="chilango_instructor_dataset_1000.jsonl",
        help="Archivo JSONL con datos de entrenamiento",
    )
    parser.add_argument("--output_dir", type=str, default="./lora-tutor")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--no_quantize", action="store_true", help="No usar cuantización 8-bit")
    return parser.parse_args()


def load_model_and_tokenizer(model_name, use_quantization):
    """Carga el modelo base y tokenizer."""
    print(f"Cargando tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    has_cuda = torch.cuda.is_available()

    if use_quantization and has_cuda:
        print("Cargando modelo en 4-bit (cuantización habilitada)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        if not has_cuda:
            print("AVISO: No se detectó GPU. Entrenando en CPU (será lento).")
        print("Cargando modelo sin cuantización")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if not has_cuda else torch.float16,
            device_map="auto" if has_cuda else None,
        )

    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, lora_r, lora_alpha):
    """Aplica adaptadores LoRA al modelo."""
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def prepare_dataset(dataset_path, tokenizer, max_length):
    """Carga y tokeniza el dataset."""
    print(f"Cargando dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path)

    def tokenize(example):
        text = (
            f"Instrucción: {example['instruction']}\n"
            f"Respuesta: {example['output']}{tokenizer.eos_token}"
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
    )
    print(f"Dataset tokenizado: {len(tokenized['train'])} ejemplos")
    return tokenized


def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_quantization=not args.no_quantize,
    )

    model = apply_lora(model, args.lora_r, args.lora_alpha)

    tokenized = prepare_dataset(args.dataset, tokenizer, args.max_length)

    has_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=has_cuda,
        bf16=False,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    print("\n--- Iniciando entrenamiento ---")
    trainer.train()

    print(f"\nGuardando adaptadores LoRA en: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
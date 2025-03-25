import os
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import opacus
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import argparse

# constants 
# required delta hyperparameter computed dynamically based on the dataset size below
MODEL_ID = "xlm-roberta-base"
LANGUAGE = "en"
BATCH_SIZE = 32
MAX_PHYSICAL_BATCH_SIZE = 8
EPOCHS = 3
LOGGING_INTERVAL = 5000
EPSILON = 7.5
MAX_GRAD_NORM = 0.1

def load_config(model_id):
    return AutoConfig.from_pretrained(model_id, num_labels=3)

def load_model(model_id, config):
    return AutoModelForSequenceClassification.from_pretrained(model_id, config=config)

def create_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, do_lower_case=False)

def freeze_layers(model):
    # only freeze last encoder layer and classifier layer(linear)
    # total params: 278,045,955
    # trainable params: 7,680,771

    trainable_layers = [model.roberta.encoder.layer[-1], model.classifier]
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()
    
    for l in trainable_layers:
        for p in l.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")
    
    return model

def process_data(examples, tokenizer, max_seq_length=128):
    return tokenizer(
        examples["premise"], 
        examples["hypothesis"], 
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )

def prepare_data(tokenizer, language):
    xnli_loaded = load_dataset("xnli", language)
    xnli_datasets = {}
    for split in ["train", "validation", "test"]:
        xnli_data = xnli_loaded[split].map(lambda x: process_data(x, tokenizer), batched=True)
        xnli_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        xnli_datasets[split] = xnli_data

    return xnli_datasets["train"], xnli_datasets["validation"], xnli_datasets["test"]

def accuracy(preds, labels):
    return (preds == labels).mean()

def train(model, train_dataloader, test_dataloader, optimizer, device, privacy_engine, criterion, delta, epochs, log_interval, max_batch_size):
    for epoch in range(1, epochs+1):
        losses = []
        with BatchMemoryManager(
            data_loader=train_dataloader, max_physical_batch_size=max_batch_size, optimizer=optimizer
        ) as memory_safe_data_loader:
            for step, batch in enumerate(tqdm(memory_safe_data_loader)):
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs = {'labels': batch['label'], 'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
                outputs = model(**inputs) # outputs = loss, logits, hidden_states, attentions
                loss = outputs[0]
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if step > 0 and step % log_interval == 0:
                    train_loss = np.mean(losses)
                    eps = privacy_engine.get_epsilon(delta)
                    eval_loss, eval_accuracy = evaluate(model, test_dataloader, device)
                    print(f"Epoch: {epoch} | Step: {step} | Train loss: {train_loss:.3f} | Eval loss: {eval_loss:.3f} | Eval accuracy: {eval_accuracy:.3f} | ɛ: {eps:.2f}")
        

def evaluate(model):
    model.eval()

    loss_list = []
    accuracy_list = []

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            inputs = {'labels': batch['label'],'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            outputs = model(**inputs)
            loss, logits = outputs[:2]

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = inputs['labels'].detach().cpu().numpy()
            loss_list.append(loss.item())
            accuracy_list.append(accuracy(preds, labels))

    model.train()
    return np.mean(loss_list), np.mean(accuracy_list)

def compute_delta(train_dataloader):
    return 1 / len(train_dataloader)

def initialize_privacy_engine(model, optimizer, train_dataloader, privacy_mode, delta):
    privacy_engine = PrivacyEngine()

    criterion = nn.CrossEntropyLoss(reduction="mean") if privacy_mode == "ghost-clipping" else None
    
    outputs = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        **({"criterion": criterion} if criterion else {}),
        target_delta=delta,
        target_epsilon=EPSILON,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        grad_sample_mode="ghost" if privacy_mode == "ghost-clipping" else "hooks",  
    )
    if privacy_mode == "ghost-clipping":
        model, optimizer, criterion, new_train_dataloader = outputs
    else:
        model, optimizer, new_train_dataloader = outputs
        criterion = None

    return model, optimizer, new_train_dataloader, criterion, privacy_engine

def parse_command_line_args():
    parser = argparse.ArgumentParser(description="Run DP RoBERTa with Vanilla DP-SGD or Ghost Clipping")
    parser.add_argument("--privacy_mode", type=str, default="vanilla", choices=["vanilla", "ghost-clipping"],
                        help="Choose between Vanilla DP-SGD and Ghost Clipping")
    args = parser.parse_args()
    print(f"Privacy Mode: {args.privacy_mode}")
    return args

if __name__ == "__main__":
    args = parse_command_line_args()    
    
    config = load_config(MODEL_ID)
    tokenizer = create_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID, config)
    model = freeze_layers(model)

    train_set, _, test_set = prepare_data(tokenizer, language=LANGUAGE)

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train() # required for initializing privacy engine
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8)

    delta = compute_delta(train_dataloader)

    model, optimizer, train_dataloader, criterion, privacy_engine = initialize_privacy_engine(model, optimizer, train_dataloader, args.privacy_mode, delta)

    train(model, train_dataloader, test_dataloader, optimizer, device, privacy_engine, criterion, delta=delta, epochs=EPOCHS, log_interval=LOGGING_INTERVAL, max_batch_size=MAX_PHYSICAL_BATCH_SIZE)
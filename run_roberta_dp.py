import os
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import re
import json
from tqdm import tqdm
import opacus
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


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

def process_data(examples, tokenizer):
    return tokenizer(
        examples["premise"], 
        examples["hypothesis"], 
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

def prepare_data(tokenizer, language):

    xnli_data = load_dataset("xnli", language)

    train_dataset = xnli_data["train"]
    dev_dataset = xnli_data["validation"]
    test_dataset = xnli_data["test"]
    
    train_dataset = train_dataset.map(lambda x: process_data(x, tokenizer), batched=True)
    dev_dataset = dev_dataset.map(lambda x: process_data(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: process_data(x, tokenizer), batched=True)
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, dev_dataset, test_dataset

def accuracy(preds, labels):
    return (preds == labels).mean()

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

if __name__ == "__main__":

    # fix data loading 
    # define train function
    # define cli function where you keep all arguments
    # keep constants in one place

    # constants
    MODEL_ID = "xlm-roberta-base"
    MAX_SEQ_LENGTH = 128
    LANGUAGE = "en"
    BATCH_SIZE = 32
    MAX_PHYSICAL_BATCH_SIZE = 8
    
    config = load_config(MODEL_ID)
    tokenizer = create_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID, config)

    train_set, dev_set, test_set = prepare_data(tokenizer, language=LANGUAGE)

    model = freeze_layers(model)

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=BATCH_SIZE)
    # for i in range(5):
    #    batch = next(iter(test_dataloader))
    #    print(batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8)

    # constants for training
    EPOCHS = 3
    LOGGING_INTERVAL = 5000
    EPSILON = 7.5
    DELTA = 1 / len(train_dataloader)
    MAX_GRAD_NORM = 0.1

    privacy_engine = PrivacyEngine()

    model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_delta=DELTA,
        target_epsilon=EPSILON,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in range(1, EPOCHS+1):
        losses = []

        with BatchMemoryManager(
            data_loader=train_dataloader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            
            for step, batch in enumerate(tqdm(memory_safe_data_loader)):
                optimizer.zero_grad()

                batch = {k: v.to(device) for k, v in batch.items()}
                inputs = {'labels': batch['label'],'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
                outputs = model(**inputs) # outputs = loss, logits, hidden_states, attentions

                loss = outputs[0]
                loss.backward()
                losses.append(loss.item())

                optimizer.step()

                if step > 0 and step % LOGGING_INTERVAL == 0:
                    train_loss = np.mean(losses)
                    eps = privacy_engine.get_epsilon(DELTA)

                    eval_loss, eval_accuracy = evaluate(model)

                    print(
                        f"Epoch: {epoch} | "
                        f"Step: {step} | "
                        f"Train loss: {train_loss:.3f} | "
                        f"Eval loss: {eval_loss:.3f} | "
                        f"Eval accuracy: {eval_accuracy:.3f} | "
                        f"ɛ: {eps:.2f}"
                    )
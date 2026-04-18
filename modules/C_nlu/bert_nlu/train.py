import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from .model import JointIntentSlotModel
from .dataset import JointNLUDataset
from .labels import INTENTS, slot_labels
from .config import (
    MODEL_NAME, MAX_LEN, BATCH_SIZE, LEARNING_RATE, EPOCHS, 
    MODEL_SAVE_PATH, LABEL_MAPS_PATH, BASE_DIR
)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Load data
    train_path = os.path.join(BASE_DIR, "data", "train.json")
    val_path = os.path.join(BASE_DIR, "data", "val.json")
    
    train_dataset = JointNLUDataset(train_path, MODEL_NAME, MAX_LEN)
    val_dataset = JointNLUDataset(val_path, MODEL_NAME, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = JointIntentSlotModel(MODEL_NAME, len(INTENTS), len(slot_labels))
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    best_val_loss = float("inf")
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_labels"].to(device)
            slot_labels_batch = batch["slot_labels"].to(device)
            
            _, _, loss = model(input_ids, attention_mask, intent_labels, slot_labels_batch)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                intent_labels = batch["intent_labels"].to(device)
                slot_labels_batch = batch["slot_labels"].to(device)
                
                _, _, loss = model(input_ids, attention_mask, intent_labels, slot_labels_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved!")
            
    # Save label maps as well for inference
    torch.save({"intents": INTENTS, "slot_labels": slot_labels}, LABEL_MAPS_PATH)

if __name__ == "__main__":
    train()

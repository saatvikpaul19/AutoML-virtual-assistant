import torch
import numpy as np
from seqeval.metrics import classification_report as seq_report
from sklearn.metrics import classification_report as sk_report
from .labels import id2intent, id2slot

def evaluate(model, data_loader, device):
    model.eval()
    
    intent_preds = []
    intent_true = []
    slot_preds = []
    slot_true = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_labels"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            
            intent_logits, slot_logits, _ = model(input_ids, attention_mask)
            
            # Intent predictions
            intent_preds.extend(torch.argmax(intent_logits, dim=1).cpu().numpy())
            intent_true.extend(intent_labels.cpu().numpy())
            
            # Slot predictions
            s_preds = torch.argmax(slot_logits, dim=2).cpu().numpy()
            s_true = slot_labels.cpu().numpy()
            
            for p, t in zip(s_preds, s_true):
                p_items = []
                t_items = []
                for pred_id, true_id in zip(p, t):
                    if true_id != -100:
                        p_items.append(id2slot[pred_id])
                        t_items.append(id2slot[true_id])
                slot_preds.append(p_items)
                slot_true.append(t_items)
                
    # Reports
    print("\n--- Intent Classification Report ---")
    print(sk_report(intent_true, intent_preds, target_names=[id2intent[i] for i in sorted(np.unique(intent_true))]))
    
    print("\n--- Slot Filling Report ---")
    print(seq_report(slot_true, slot_preds))

    return intent_preds, intent_true, slot_preds, slot_true

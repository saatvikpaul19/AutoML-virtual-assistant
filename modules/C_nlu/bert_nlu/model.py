import torch
import torch.nn as nn
from transformers import AutoModel

class JointIntentSlotModel(nn.Module):
    def __init__(self, model_name, num_intents, num_slots, dropout=0.1):
        super(JointIntentSlotModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Intent classification head
        self.intent_classifier = nn.Linear(self.encoder.config.hidden_size, num_intents)
        
        # Slot classification head
        self.slot_classifier = nn.Linear(self.encoder.config.hidden_size, num_slots)
        
    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        # Encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # [batch, seq_len, hidden]
        
        # Intent head: Use CLS token (first token)
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        intent_logits = self.intent_classifier(cls_output)
        
        # Slot head: Use all tokens
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        loss = 0
        if intent_labels is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, intent_logits.size(-1)), intent_labels.view(-1))
            loss += intent_loss
            
        if slot_labels is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, slot_logits.size(-1))
                active_labels = torch.where(
                    active_loss, slot_labels.view(-1), torch.tensor(slot_loss_fct.ignore_index).type_as(slot_labels)
                )
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1))
            loss += slot_loss
            
        return intent_logits, slot_logits, loss

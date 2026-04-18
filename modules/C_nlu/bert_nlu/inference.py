import torch
from transformers import AutoTokenizer
from .model import JointIntentSlotModel
from .labels import id2intent, id2slot, INTENTS, slot_labels
from .decode_slots import decode_slots
from .config import MODEL_NAME, MODEL_SAVE_PATH, MAX_LEN
import torch.nn.functional as F

class BERTNLUInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load model metadata (labels)
        # Note: In production we'd load this from label_maps.pt, but labels.py is consistent here
        self.model = JointIntentSlotModel(MODEL_NAME, len(INTENTS), len(slot_labels))
        
        if torch.os.path.exists(MODEL_SAVE_PATH):
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

    def parse(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(device=self.device)
        
        with torch.no_grad():
            intent_logits, slot_logits, _ = self.model(input_ids, attention_mask)
            
            # Intent
            intent_probs = F.softmax(intent_logits, dim=1)
            intent_conf, intent_id = torch.max(intent_probs, dim=1)
            predicted_intent = id2intent[intent_id.item()]
            
            # Slots
            slot_ids = torch.argmax(slot_logits, dim=2).squeeze().cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze())
            
            extracted_slots = decode_slots(tokens, slot_ids)
            
            # token_slots structure for requested output
            token_slots = []
            for t, s_id in zip(tokens, slot_ids):
                if t not in ["[CLS]", "[SEP]", "[PAD]"]:
                    token_slots.append({"token": t, "label": id2slot[s_id]})
                    
        return {
            "intent": predicted_intent,
            "intent_confidence": float(intent_conf.item()),
            "slots": extracted_slots,
            "token_slots": token_slots,
            "fallback_used": False
        }

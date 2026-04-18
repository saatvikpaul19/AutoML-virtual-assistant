import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from .labels import intent2id, slot2id

class JointNLUDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_len):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        slot_labels = item["slot_labels"]
        intent = item["intent"]
        
        # Encoding with word split info
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Alignment of labels
        word_ids = encoding.word_ids() # Map tokens to original words
        aligned_slot_labels = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS], [SEP], [PAD]
                aligned_slot_labels.append(-100)
            else:
                # Get the label for this word
                label = slot_labels[word_idx]
                label_id = slot2id.get(label, slot2id["O"])
                
                # Only label the first subword token
                # This check ensures only the first piece of a split word gets the label, others get -100
                # But wait, logic: if word_idx is the same as previous, it's a subword piece.
                # Actually, word_ids() returns the same index for all pieces of a word.
                # We want to label the first one B- or I- and subsequent subwords -100 to ignore them.
                # Let's verify if we should use -100 for subwords as per user request.
                pass
        
        # Correctly aligning subword labels
        final_slot_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                final_slot_ids.append(-100)
            elif word_idx != prev_word_idx:
                # First subword of the word
                final_slot_ids.append(slot2id.get(slot_labels[word_idx], slot2id["O"]))
            else:
                # Subsequent subwords
                final_slot_ids.append(-100)
            prev_word_idx = word_idx
            
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "intent_labels": torch.tensor(intent2id[intent], dtype=torch.long),
            "slot_labels": torch.tensor(final_slot_ids, dtype=torch.long)
        }

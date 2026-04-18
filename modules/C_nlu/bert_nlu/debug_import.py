import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    print("Testing Transformers import...")
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    from transformers import AutoModel, AutoTokenizer
    print("Loading distilbert-base-uncased...")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    print("Model loaded successfully!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

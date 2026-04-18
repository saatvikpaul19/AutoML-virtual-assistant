from huggingface_hub import InferenceClient

import os

# The token provided by the user
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Using Qwen 2.5 72B Instruct via Serverless Inference API
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    return _client

def query_qwen(system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to the Qwen LLM using HF InferenceClient."""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = _get_client().chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Qwen API error: {e}")
        # Soft fallback
        return f"I'm sorry, I couldn't reach my brain to answer that right now. (Error: {str(e)[:50]})"

class QwenAssistant:
    @staticmethod
    def format_best_accuracy(user_name: str, accuracy_val: float) -> str:
        """Have Qwen format the accuracy result to speak aloud."""
        sys = "You are a friendly AI voice assistant called Mycroft. Keep it to 1 sentence, warm, engaging."
        user = f"Address the user '{user_name}'. Say these are the results and the best accuracy was {accuracy_val:.4f}."
        return query_qwen(sys, user)
        
    @staticmethod
    def suggest_model(dataset_profile: dict, is_tabular: bool) -> str:
        """Have Qwen suggest a model based on tabular profile."""
        sys = "You are an expert Machine Learning AI assistant called Mycroft. Provide a very short 1-2 sentence recommendation for a baseline ML model. Be concise."
        user = f"Here is the dataset profile: {dataset_profile}. It is tabular: {is_tabular}. Don't use pip installs. What is a good baseline model?"
        
        # Try LLM
        ans = query_qwen(sys, user)
        if "couldn't reach my brain" in ans:
            # Hardcoded fallback if no API key
            if is_tabular:
                return "Based on your data, I recommend starting with XGBoost or a Random Forest Classifier as a strong baseline."
            else:
                return "For this data type, you might want to consider a standard Deep Learning architecture like ResNet or BERT depending on the modality."
        return ans
        
    @staticmethod
    def suggest_hyperparameters(model_name: str, dataset_profile: dict) -> str:
        """Have Qwen suggest hyperparameters."""
        sys = "You are an expert ML tuning assistant called Mycroft. Give a quick 1-sentence suggestion of default parameters."
        user = f"Dataset profile: {dataset_profile}. Suggest default hyperparameters like Learning Rate, Epochs, Batch Size for {model_name}."
        
        ans = query_qwen(sys, user)
        if "couldn't reach my brain" in ans:
            return f"For {model_name}, a good starting point is a learning rate of 0.01, 20 epochs, and a batch size of 32."
        return ans
        
    @staticmethod
    def format_dataset_changes(changes: list) -> str:
        """Format the summary of dataset cleaning changes for TTS."""
        if not changes:
            return "No changes were made to the dataset."
        sys = "You are a helpful AI data scientist called Mycroft. Summarize these dataset changes in 1 fluid sentence."
        user = f"Changes made: {changes}"
        return query_qwen(sys, user)

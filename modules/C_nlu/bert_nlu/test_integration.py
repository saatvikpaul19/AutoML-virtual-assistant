from modules.C_nlu.nlu_pipeline import understand

def test_pipeline():
    print("Testing NLU Pipeline Integration...")
    
    # Since model might not be trained yet due to memory, 
    # the 'understand' function should automatically fall back to Regex properly.
    
    samples = [
        "hey mycroft load mnist dataset",
        "set learning rate to 0.05",
        "what is the weather in Ottawa",
        "tell me the results"
    ]
    
    for s in samples:
        print(f"\nUtterance: {s}")
        result = understand(s)
        print(f"Intent: {result['intent']}")
        print(f"Slots: {result['slots']}")
        print(f"Fallback Used: {result['fallback_used']}")

if __name__ == "__main__":
    test_pipeline()

def generate_stateless_response(intent: str, result: dict) -> str:
    if not result.get("success"):
        return result.get("error", "The Kaggle request could not be completed.")

    if intent == "search_dataset":
        return f"Found {len(result.get('results', []))} Kaggle dataset result(s) for '{result.get('query', '')}'."

    if intent == "get_dataset_info":
        return f"Retrieved Kaggle dataset information for '{result.get('query', result.get('dataset_query', 'your query'))}'."

    if intent == "show_competition":
        return f"Retrieved {len(result.get('results', []))} Kaggle competition result(s)."

    if intent == "show_leaderboard":
        return f"Retrieved Kaggle leaderboard for '{result.get('competition_name', 'the selected competition')}'."

    return "Kaggle request completed successfully."
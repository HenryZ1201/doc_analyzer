import json

def save_list_of_dicts_to_json(data, file_path):
    """
    Save a list of dictionaries to a JSON file.

    Args:
        data (list): List of dictionaries.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
import pandas as pd
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def filter_rows_with_names_and_places(df, text_column):
    """
    Filters out rows in a pandas DataFrame that contain names of people or places.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - text_column (str): The name of the column containing text to check for named entities.

    Returns:
    - pd.DataFrame: A DataFrame with rows containing names of people or places removed.
    """

    # Define a function to check if a text contains person or place names
    def contains_person_or_place(text):
        doc = nlp(text)
        return any(ent.label_ in ["PERSON", "GPE", "LOC"] for ent in doc.ents)

    # Apply the check function to each row and filter the DataFrame
    filtered_df = df[~df[text_column].apply(contains_person_or_place)]
    return filtered_df

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'text': ["Alice went to Paris", "The sky is blue.", "John Doe is a software engineer from New York."]
    }
    df = pd.DataFrame(data)

    # Filter the DataFrame
    filtered_df = filter_rows_with_names_and_places(df, "text")
    
    print(filtered_df)
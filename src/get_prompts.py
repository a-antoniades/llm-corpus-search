import openai

# Your API key from OpenAI
api_key = "YOUR_API_KEY"

def generate_text(prompt, model="davinci", max_tokens=50):
    """
    Sends a prompt to the OpenAI GPT API to generate text.
    
    Parameters:
        prompt (str): The prompt you want to send.
        model (str): The model you want to use (e.g. "davinci", "curie").
        max_tokens (int): The maximum number of tokens to generate.
        
    Returns:
        str: The generated text.
    """
    try:
        # Send the prompt to the OpenAI GPT API
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens
        )
        
        # Extract the generated text from the response
        generated_text = response.choices[0].text
        
        return generated_text
    
    except Exception as e:
        return str(e)

# Example usage:
prompt = "Once upon a time in a faraway land,"
generated_text = generate_text(prompt)
print(generated_text)
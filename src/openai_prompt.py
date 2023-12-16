import openai

openai.api_key = "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"

NUM_CHOICES = 5

prompt = "Explain large language models to me"
response = openai.Completion.create(
    engine="text-davinci-003",  # Choose the appropriate GPT model
    prompt=prompt,
    temperature=1,
    max_tokens=1000,
    n=NUM_CHOICES,
    stop=None,
)

# Extract the generated response from the API response
for i in range(NUM_CHOICES):
    generated_text = response.choices[0].text.strip()
    # Print the generated response
    print("Generated Choice: {}".format(i + 1))
    print(generated_text)
    print("\n")

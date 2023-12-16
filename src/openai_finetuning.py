import time
import json
import os
import openai

openai.api_key = "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"
response_id = "ft-ydiAlagLGUK2kZ4qAXQ4tLCi"

PARA_SPLIT_TOKENS = 10

dissertation_file = "file-lP562UdlGJLGlXFgOV4kVJDz"

FILE = dissertation_file

# model_engine = "text-davinci-003"
# n_epochs = 3
# batch_size = 4
# learning_rate = 1e-5
# max_tokens = 1024

# fine_tune_response = openai.FineTune.create(training_file=FILE, model="davinci")
# print(fine_tune_response)

# fine_tune_events = openai.FineTune.list(id=fine_tune_response.id)

response = openai.FineTune.retrieve(id=response_id)
# print(response)
model = response.fine_tuned_model
# print(model)


new_prompt = "What is the robot flange->"

answer = openai.Completion.create(
    model=model, prompt=new_prompt, max_tokens=1000, temperature=0.9
)
print(answer["choices"][0]["text"])

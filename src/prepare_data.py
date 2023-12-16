import re
import os
import math
import json
import random
import openai
import subprocess
from IPython import embed
from openai import Completion
from alive_progress import alive_bar

PREPARE = True
UPLOAD = False
EDIT = False
FILE = "../data/dissertation_raw.txt"
PREPARED_FILE = "../data/dissertation_prepared.txt"
JSON_FILE = "../data/dissertation.jsonl"
edited_FILE = "../data/dissertation_edited.jsonl"

openai.api_key = "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"


def extract_numbered_lines(text):
    lines = text.split("\n")
    numbered_lines = []

    # Regular expression pattern to match lines with numbered index
    pattern = r"^\d+\."

    for line in lines:
        if re.match(pattern, line):
            numbered_lines.append(line)

    return numbered_lines


def build_messages(messages):
    return [
        "".join(
            [
                "What are some questions that can be asked to get the following text as a response?",
                "I want only numbered questions as your response.\n",
                message,
            ]
        )
        for message in messages
    ]


# Prepare the data
# def filter_text_file(file_path):
#     # Regular expression pattern to match words
#     word_pattern = r"\b\w+\b"

#     # List to store filtered words and paragraphs
#     filtered_words = []
#     filtered_paragraphs = []

#     # Read the text file
#     with open(file_path, "r") as file:
#         # Read line by line
#         for line in file:
#             # Remove leading/trailing whitespaces
#             line = line.strip()

#             # Skip empty lines
#             if not line:
#                 continue

#             words = re.findall(word_pattern, line)
#             filtered_words.extend(words)

#             # Check if the line is a paragraph (ending with a period)
#             if line[-1] != "\n":
#                 if len(filtered_words) > 20:
#                     filtered_paragraphs.append(" ".join(filtered_words))
#                 filtered_words = []

#     return filtered_paragraphs


def filter_text_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # # Remove LaTeX commands and other unwanted characters
    # cleaned_content = re.sub(
    #     r"\\[^\\]*\{[^\}]*\}", "", content
    # )  # Remove LaTeX commands
    # cleaned_content = re.sub(
    #     r"\\[^\\]*", "", cleaned_content
    # )  # Remove remaining backslashes
    cleaned_content = re.sub(
        r"\$.*?\$", "", content
    )  # Remove equations enclosed in '$'
    cleaned_content = re.sub(
        r"\\begin\{figure\}.*?\\end\{figure\}", "", cleaned_content, flags=re.DOTALL
    )  # Remove figures
    cleaned_content = re.sub(
        r"\\begin\{algorithm\}.*?\\end\{algorithm\}", "", content, flags=re.DOTALL
    )  # Remove algorithm
    cleaned_content = re.sub(
        r"\\begin\{table\}.*?\\end\{table\}", "", cleaned_content, flags=re.DOTALL
    )  # Remove tables
    cleaned_content = re.sub(
        r"\\begin\{eqnarray\}.*?\\end\{eqnarray\}",
        "",
        cleaned_content,
        flags=re.DOTALL,
    )  # Remove equation array
    cleaned_content = re.sub(
        r"\\section\{.*?\}", "", cleaned_content
    )  # Remove section labels

    # Extract paragraphs with complete sentences
    paragraphs = cleaned_content.split("\n\n")
    complete_paragraphs = [p.strip() for p in paragraphs if re.search(r"[.!?]", p)]

    return complete_paragraphs


def write_paragraphs_to_file(paragraphs, output_file):
    with open(output_file, "w") as file:
        for i, paragraph in enumerate(paragraphs, start=1):
            if len(paragraph) < 100:
                continue
            file.write(paragraph + "\n\n")


if PREPARE:
    # Call the function to filter the text file
    filtered_paragraphs = filter_text_file(os.path.abspath(FILE))

    write_paragraphs_to_file(filtered_paragraphs, PREPARED_FILE)

    print("Sending question generation request to OpenAI")

    data = []
    counter = 0
    MAX_COUNT = 500
    with alive_bar(MAX_COUNT) as bar:
        while counter < MAX_COUNT:
            paragraph = random.choice(filtered_paragraphs)
            response = openai.Completion.create(
                engine="text-davinci-003",  # Choose the appropriate GPT model
                prompt=build_messages([paragraph])[0],
                temperature=0.7,
                max_tokens=1000,
                n=1,
                stop=None,
            )
            prompt = extract_numbered_lines(response.choices[0].text.strip())
            completion = paragraph
            try:
                question = random.choice(prompt)
                question = re.sub(r"^\d+\.\s", "", question)
                data.append(
                    {
                        "prompt": "".join([question, "->"]),
                        "completion": "".join([completion, "\n"]),
                    }
                )
                counter += 1
            except:
                pass
            bar()

    with open(JSON_FILE, "w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

if EDIT:
    data = []
    with open(JSON_FILE, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))

    random.shuffle(data)

    with open(edited_FILE, "w") as file:
        for item in data:
            item["prompt"] = re.sub(r"\d+\.\s", "", item.get("prompt"))
            json.dump(item, file)
            file.write("\n")

# def bash(command):
#     process = subprocess.Popen(
#         command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#     )
#     output, error = process.communicate()
#     # Print the command error, if any
#     if error:
#         print("Command Error:")
#         print(error.decode())


# command = "openai tools fine_tunes.prepare_data -f ../data/rcim2021_training.json"
# bash(command)
# command = "openai tools fine_tunes.prepare_data -f ../data/rcim2021_validation.json"
# bash(command)

# upload_response = openai.File.create(
#     file=open("../data/rcim2021_training_prepared.jsonl", "rb"), purpose="fine-tune"
# )
# file_id = upload_response.id
# print("Training file ID: ", upload_response)

# upload_response = openai.File.create(
#     file=open("../data/rcim2021_validation_prepared.jsonl", "rb"), purpose="fine-tune"
# )
# file_id = upload_response.id
# print("Validation file ID: ", upload_response)

if UPLOAD:
    upload_response = openai.File.create(
        file=open("../data/dissertation.jsonl", "rb"),
        purpose="fine-tune",
    )
    file_id = upload_response.id
    print("Data file ID: ", upload_response)

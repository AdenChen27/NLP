import os
import csv
import json
import time
import nltk
import openai
import pandas as pd
from collections import Counter


API_KEY = ""

topic_labels = [
    "Education Policy", 
    "Personal Life", 
    "Budget and Funding", 
    "School Safety", 
    "Community Engagement", 
    "Curriculum Updates", 
    "Technology Integration", 
    "Student Achievement", 
    "Equity and Inclusion", 
    "Infrastructure and Facilities", 
    "Government Interaction"
]

sentiment_labels = ["anger", "happiness", "frustration", "sadness", "disgust", "trust"]



def __prepare_prompt(
        text, 
        labels=(topic_labels, sentiment_labels), 
        prompt_filename="gpt-prompt-combined"
    ):
    """
    text: relevant text article
    labels: should be a list of strings
    prompt_filename: "gpt-prompt-topic", "gpt-prompt-sentiment", or "gpt-prompt-combined"
        - When using prompts "gpt-prompt-combined", pass both topic and 
        sentiment labels as (topic_labels, sentiment_labels)
        - Otherwise, pass relevant labels only (there should be only one list). 
    """

    with open(prompt_filename, 'r') as file:
        base_prompt = file.read()

    prompt = base_prompt.replace("[STATEMENT]", text)

    if type(labels) is tuple: 
        # two label lists
        if len(labels) != 2:
            raise "check labels tuple len"
        labels1, labels2 = labels
        prompt = prompt.replace("[LABELS1]", ", ".join(labels1))
        prompt = prompt.replace("[LABELS2]", ", ".join(labels2))
    else:
        # one label lists
        prompt = prompt.replace("[LABELS]", ", ".join(labels))
    return prompt


def __make_api_call(client, model, system_content, prompt, temperature):
    while True:
        #Make API call
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
                ],
                temperature = temperature
            )
            response = chat_completion.choices[0].message.content
            #Verify clean output
            try:
                response_clean = json.loads(response)
            except (json.decoder.JSONDecodeError) as e:
                print(' Incorrect JSON format. Retrying...')
                continue    #retry API call if JSON parsing fails
            break  # Break out of the loop if API call is successful
        except openai.error.APIError as e:
            print(" API error. Retrying..."\
                  "(if error persists, check status.openai.com)")
            time.sleep(10)
        except openai.error.Timeout as e:
            print(" Request timed out. Retrying... "\
                  "(if error persists, check internet connection)")
            time.sleep(5)
        except openai.error.RateLimitError as e:
            print(" Reached rate limit. Retrying... "\
                  "(if error persists, check number of tokens/requests)")
            time.sleep(60)
        except openai.error.APIConnectionError as e:
            print(" API connection error. Retrying... "\
                  "(if error persists, check network/proxy config/ssl/firewall)")
            time.sleep(5)
        except openai.error.InvalidRequestError as e:
            print(" Invalid request error. Retrying... "\
                  "(if error persists, check for invalid/missing request parameters)")
        except openai.error.AuthenticationError as e:
            print(" Authentication error. Retrying... "\
                  "(if error persists, check for invalid/expired/revoked API key or token)")
        except openai.error.ServiceUnavailableError as e:
            print(" Server is overloaded. Retrying... "\
                  "(if error persists, check status.openai.com)")
            time.sleep(120)
        
    return response_clean


def analyze_text(
        text, 
        labels=(topic_labels, sentiment_labels), 
        prompt_filename="gpt-prompt-combined"
    ):
    """
    text: relevant text article
    labels: should be a list of strings
    prompt_filename: "gpt-prompt-topic", "gpt-prompt-sentiment", or "gpt-prompt-combined"
        - When using prompts "gpt-prompt-combined", pass both topic and 
            sentiment labels as (topic_labels, sentiment_labels)
        - Otherwise, pass relevant labels only (there should be only one list). 
    """
    # API parameters
    model = 'gpt-4o'
    system_content = 'You are a knowledgeable and unbiased judge.'
    n_groups=None
    temperature = 0

    # https://platform.openai.com/docs/guides/text-generation/json-mode
    prompt = __prepare_prompt(text, labels, prompt_filename)
    # print(prompt)
    # return 
    client = openai.OpenAI(
        # This is the default and can be omitted
        api_key=API_KEY,
    )

    return __make_api_call(client, model, system_content, prompt, temperature) 


def check(rst, labels=(topic_labels, sentiment_labels)):
    # check if return values by `analyze_text` is valid
    # labels: when using `combined` prompt, pass both topic and sentiment labels; see above.

    if type(labels) is tuple:
        # two label lists
        labels = labels[0] + labels[1]

    labels = set(labels)
    if (rst.keys() - labels) != set() or (labels - rst.keys()) != set():
        return False
    for v in rst.values():
        if v < 0 or v > 1:
            return False
    return True





import openai
from retry import retry

def initialize_client():
    openai.api_key = "ADD_YOUR_KEY_HERE"  # Replace with your OpenAI API key
    return openai

@retry(tries=5, delay=1, backoff=1, jitter=(0, 5), max_delay=10)
def call_chatgpt(client, chatgpt_messages, max_tokens=40, model="gpt-4o-2024-08-06"): 
    response = client.chat.completions.create(
        model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens
    )
    reply = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    return reply, total_tokens

def prepare_chatgpt_message(prompt):
    system_message = "an evaluator who rates my answer based on the correct answer"
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": "{}".format(prompt)})
    return messages

def gpt_forward(data):
    try:
        answer, GT = data

        # Initialize the client inside the worker process
        client = initialize_client()

        prompts = ("Rate my answer based on the correct answer out of 100, with higher "
                "scores indicating that the answer is closer to the correct answer. "
                "Just rate the similarity of the content not the sentence structure. "
                "If the content is completely different rate with 0. You should be accurate to single digits like 62, 78, 41, etc. "
                "Output the number only. This is the correct answer: " + GT + 
                " This is my answer: " + answer)

        messages = prepare_chatgpt_message(prompts)
        reply, total_tokens = call_chatgpt(client, messages, max_tokens=3000)
    except Exception as e:
        print(f"Error: {e}")
        return -1
    return int(reply)

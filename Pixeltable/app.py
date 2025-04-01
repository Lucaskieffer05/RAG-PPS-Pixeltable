import pixeltable as pxt
from pixeltable.functions import llama_cpp

pxt.drop_dir('llama_demo', force=True)
pxt.create_dir('llama_demo')

t = pxt.create_table('llama_demo.chat', {'input': pxt.String})

# Add a computed column that uses llama.cpp for chat completion
# against the input.

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': t.input}
]

t.add_computed_column(result=llama_cpp.create_chat_completion(
    messages,
    repo_id='Qwen/Qwen2.5-0.5B-Instruct-GGUF',
    repo_filename='*q5_k_m.gguf',
    temperature=0.7,
    max_new_tokens=256,
    top_p=0.95,
    top_k=40,
    n=1,
))

# Extract the output content from the JSON structure returned
# by llama_cpp.

t.add_computed_column(output=t.result.choices[0].message.content)
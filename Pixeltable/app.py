import ollama

ollama.pull('qwen2.5:0.5b')
ollama.generate('qwen2.5:0.5b', 'Cual es la capital de buenos aires?')['response']

import pixeltable as pxt
from pixeltable.functions.ollama import chat

pxt.drop_dir('ollama_demo', force=True)
pxt.create_dir('ollama_demo')
t = pxt.create_table('ollama_demo.chat', {'input': pxt.String})

messages = [{'role': 'user', 'content': t.input}]

t.add_computed_column(output=chat(
    messages=messages,
    model='qwen2.5:0.5b',
    # These parameters are optional and can be used to tune model behavior:
    options={'max_tokens': 300, 'top_p': 0.9, 'temperature': 0.5},
))

# Extract the response content into a separate column

t.add_computed_column(response=t.output.message.content)
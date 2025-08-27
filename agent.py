import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
from multiwoz_utils.database import default_database

# ========== Ontology Tool ==========
@register_tool('ontology_lookup')
class OntologyLookupTool(BaseTool):
    description = 'Ontology lookup tool. Input domain and slot, return possible values for the slot. Both domain and slot should be in English and lowercase.'
    parameters = [
        {'name': 'domain', 'type': 'string', 'description': 'Domain name, e.g., restaurant, hotel', 'required': True},
        {'name': 'slot', 'type': 'string', 'description': 'Slot name, e.g., area, food', 'required': True}
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        domain = args.get('domain')
        slot = args.get('slot')
        # Read ontology.json
        with open('ontology.json', 'r', encoding='utf-8') as f:
            ontology = json.load(f)
        # Compatible with key format domain-slot
        key = f"{domain}-{slot}"
        values = ontology.get(key, [])
        return json.dumps({'domain': domain, 'slot': slot, 'values': values}, ensure_ascii=False)

# ========== MultiWOZ Database Query Tool ==========
@register_tool('multiwoz_db_query')
class MultiwozDBQueryTool(BaseTool):
    description = (
        'MultiWOZ database query tool. Input domain (e.g., hotel/restaurant) and condition (e.g., area=centre, price=cheap), return matched database entries. '
        'Example: {"domain": "hotel", "condition": "area=north, pricerange=cheap"} '
        'Return: a list of matched entities.'
    )
    parameters = [
        {'name': 'domain', 'type': 'string', 'description': 'Domain to query, e.g., hotel, restaurant', 'required': True},
        {'name': 'condition', 'type': 'string', 'description': 'Query condition, e.g., area=centre, price=cheap', 'required': False}
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        domain = args.get('domain')
        condition = args.get('condition', '')
        constraints = {}
        if condition:
            for cond in condition.split(','):
                cond = cond.strip()
                if '=' in cond:
                    k, v = cond.split('=', 1)
                    constraints[k.strip()] = v.strip()
        result = default_database.query(domain, constraints)
        return json.dumps({'result': result}, ensure_ascii=False)

# ========== Auto-generate ontology concise prompt fragment ==========
def build_ontology_prompt():
    with open('ontology.json', 'r', encoding='utf-8') as f:
        ontology = json.load(f)
    # Count all domains and slots
    domain_slots = {}
    for key in ontology:
        if '-' in key:
            domain, slot = key.split('-', 1)
            if domain not in domain_slots:
                domain_slots[domain] = []
            domain_slots[domain].append(slot)
    lines = []
    for domain, slots in domain_slots.items():
        slot_str = ', '.join(slots)
        lines.append(f"- {domain}: {slot_str}")
    return '\n'.join(lines)

# ========== Load few-shot examples ==========
def load_few_shot_examples():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_test_samples.txt')
    if not os.path.exists(file_path):
        return ''
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    # Tool call few-shot example
    tool_demo = (
        "User: I want to book a train from Cambridge to Stansted Airport on Saturday.\n"
        "Assistant: Let me check the available trains for you.\n"
        "#TOOL_CALL: multiwoz_db_query(domain='train', condition='departure=cambridge, destination=stansted airport, day=saturday')\n"
        "#TOOL_RESULT: [{\"trainid\": \"TR3128\", \"departure\": \"cambridge\", \"destination\": \"stansted airport\", \"day\": \"saturday\", \"leaveat\": \"20:40\", \"arriveby\": \"21:08\"}]\n"
        "Assistant: There is a train TR3128 leaving at 20:40 and arriving at 21:08. Would you like to book it?\n"
    )
    # Only keep first 2 multi-turn dialogues
    dialogues = content.split('\n\n')
    first_two = '\n\n'.join(dialogues[:2])
    return f"\nHere is a tool-use dialogue example (DO NOT make up any entity, always use the tool):\n{tool_demo}\nHere are real multi-turn dialogue examples between user and assistant:\n{first_two}\n"

# ========== System prompt design ==========
ontology_brief = build_ontology_prompt()
few_shot_examples = load_few_shot_examples()
system_instruction = f'''
You are a multi-domain task-oriented dialogue assistant following the MultiWOZ dataset style. You can help users book restaurants, hotels, check train schedules, call taxis, etc.
You MUST use the multiwoz_db_query tool to retrieve real entities from the database before making any recommendations. Do NOT make up any entity or information.
You can use the multiwoz_db_query tool to query the database, and the ontology_lookup tool to check possible values for domains and slots.
A typical dialogue flow includes: understanding user intent, identifying domain, extracting slots, querying the database, and generating natural language responses.
You support the following domains and slots:
{ontology_brief}
If you need to know the possible values for a domain/slot, you can call the ontology_lookup tool.
Please make decisions autonomously based on the context, call tools when necessary, and keep your responses concise and natural.
{few_shot_examples}
'''

llm_cfg = {
    'model': 'qwen:4b',
    'model_type': 'oai',
    'model_server': 'http://localhost:11434/v1',
    'api_key': 'ollama',
    # 'generate_cfg': {
        # (Optional) LLM hyperparameters:
        # This parameter affects tool call parsing logic, default False:
        # Set to True: when content is `<think>This is thinking</think>This is answer`
        # Set to False: response consists of reasoning_content and content
        # 'thought_in_content': True,

        # Tool call template: default is nous (recommended for qwen3):
        # 'fncall_prompt_type': 'nous',

        # Maximum input length, messages will be truncated if exceeded, adjust based on model API:
        # 'max_input_tokens': 512,

        # Parameters passed directly to model API, such as top_p, enable_thinking, etc., refer to API documentation:
        'top_p': 0.95,
        'temperature': 0.6,
        'top_k': 20,
        'min_p': 0,
        'enable_thinking': True,
        # Recommended output length: 32768 for most queries, 38912 for complex tasks
        'max_tokens': 32768
    # }
}

tools = ['multiwoz_db_query', 'ontology_lookup']
bot = Assistant(llm=llm_cfg, system_message=system_instruction, function_list=tools)

messages = []
if __name__ == '__main__':
    # print('===== SYSTEM PROMPT START =====')
    # print(system_instruction)
    # print('===== SYSTEM PROMPT END =====')
    # sys.exit(0)
    while True:
        query = input('\nUser: ')
        if query.strip().lower() == 'exit':
            print('Exited conversation.')
            break
        messages.append({'role': 'user', 'content': query})
        print('Agent:')
        response_plain_text = ''
        for resp in bot.run(messages=messages, stream=False):
            response_plain_text = typewriter_print(resp, response_plain_text)
        messages.append({'role': 'assistant', 'content': response_plain_text}) 
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
from multiwoz_utils.database import default_database

# ========== Ontology工具 ==========
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
        # 读取ontology.json
        with open('ontology.json', 'r', encoding='utf-8') as f:
            ontology = json.load(f)
        # 兼容key格式 domain-slot
        key = f"{domain}-{slot}"
        values = ontology.get(key, [])
        return json.dumps({'domain': domain, 'slot': slot, 'values': values}, ensure_ascii=False)

# ========== MultiWOZ数据库查询工具 ==========
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

# ========== 自动生成ontology精简prompt片段 ==========
def build_ontology_prompt():
    with open('ontology.json', 'r', encoding='utf-8') as f:
        ontology = json.load(f)
    # 统计所有domain和slot
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

# ========== 读取few-shot示例 ==========
def load_few_shot_examples():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_test_samples.txt')
    if not os.path.exists(file_path):
        return ''
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    # 工具调用few-shot示例
    tool_demo = (
        "User: I want to book a train from Cambridge to Stansted Airport on Saturday.\n"
        "Assistant: Let me check the available trains for you.\n"
        "#TOOL_CALL: multiwoz_db_query(domain='train', condition='departure=cambridge, destination=stansted airport, day=saturday')\n"
        "#TOOL_RESULT: [{\"trainid\": \"TR3128\", \"departure\": \"cambridge\", \"destination\": \"stansted airport\", \"day\": \"saturday\", \"leaveat\": \"20:40\", \"arriveby\": \"21:08\"}]\n"
        "Assistant: There is a train TR3128 leaving at 20:40 and arriving at 21:08. Would you like to book it?\n"
    )
    # 只保留前2个多轮对话
    dialogues = content.split('\n\n')
    first_two = '\n\n'.join(dialogues[:2])
    return f"\nHere is a tool-use dialogue example (DO NOT make up any entity, always use the tool):\n{tool_demo}\nHere are real multi-turn dialogue examples between user and assistant:\n{first_two}\n"

# ========== system prompt设计 ==========
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
        # (可选) LLM超参数:
        # 该参数会影响工具调用解析逻辑，默认False:
        # 设置为True: 当内容为`<think>这是思考</think>这是答案`
        # 设置为False: 响应由reasoning_content和content组成
        # 'thought_in_content': True,

        # 工具调用模板: 默认为nous（推荐qwen3）:
        # 'fncall_prompt_type': 'nous',

        # 最大输入长度，超出会截断消息，请根据模型API调整:
        # 'max_input_tokens': 512,

        # 直接传递给模型API的参数，如top_p、enable_thinking等，具体参考API说明:
        'top_p': 0.95,
        'temperature': 0.6,
        'top_k': 20,
        'min_p': 0,
        'enable_thinking': True,
        # 建议输出长度：大多数查询建议32768，复杂任务可用38912
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
            print('已退出对话。')
            break
        messages.append({'role': 'user', 'content': query})
        print('Agent:')
        response_plain_text = ''
        for resp in bot.run(messages=messages, stream=False):
            response_plain_text = typewriter_print(resp, response_plain_text)
        messages.append({'role': 'assistant', 'content': response_plain_text}) 
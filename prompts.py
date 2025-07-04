# prompts.py

# Prompt for state extraction
STATE_EXTRACTION_PROMPT_TEMPLATE = """\
Extract entity values only from the last utterance by the customer in the conversation and output them in JSON format.
You should focus exclusively on the values mentioned in the last customer utterance, and ignore any information from previous turns.

The possible fields are:
{fields_desc}

Do not extract any other fields!
If a value is not specified, leave it as an empty string.

------
{examples}
---------
Now complete the following example:
input: {input}  # This is the full dialogue history.
Customer: {customer}  # This is the last utterance from the customer. Only extract from this.
output:
state:
"""

# Prompt for response generation（分 domain）
RESTAURANT_RESPONSE_GENERATION_PROMPT = """\
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is a number of restaurants in the database currently corresponding to the user's request.

If you find a restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone], or [restaurant_postcode] if asked.
If booking, provide [reference] in the answer.
Always act as if the booking is successfully done.

------
{examples}
---------
Now complete the following example:
input: {input}
Customer: {customer}
state: {state}
database: {database}
output:
response:"""

HOTEL_RESPONSE_GENERATION_PROMPT = """\
Definition: You are an assistant that helps people to book a hotel.
You can search for a hotel by area, price, stars, parking, internet, or type.
There is a number of hotels in the database currently corresponding to the user's request.

If you find a hotel, provide [hotel_name], [hotel_address], [hotel_phone], or [hotel_postcode] if asked.
If booking, provide [reference] in the answer.
Always act as if the booking is successfully done.

------
{examples}
---------
Now complete the following example:
input: {input}
Customer: {customer}
state: {state}
database: {database}
output:
response:"""

ATTRACTION_RESPONSE_GENERATION_PROMPT = """\
Definition: You are an assistant that helps people to find an attraction.
You can search for an attraction by area or type.
There is a number of attractions in the database currently corresponding to the user's request.

If you find an attraction, provide [attraction_name], [attraction_address], [attraction_phone], or [attraction_postcode] if asked.

------
{examples}
---------
Now complete the following example:
input: {input}
Customer: {customer}
state: {state}
database: {database}
output:
response:"""

TRAIN_RESPONSE_GENERATION_PROMPT = """\
Definition: You are an assistant that helps people to book a train.
You can search for a train by departure, destination, day, leaveat, arriveby.
There is a number of trains in the database currently corresponding to the user's request.

If you find a train, provide [train_id], [departure], [destination], [leaveat], [arriveby], or [duration] if asked.
If booking, provide [reference] in the answer.
Always act as if the booking is successfully done.

------
{examples}
---------
Now complete the following example:
input: {input}
Customer: {customer}
state: {state}
database: {database}
output:
response:"""

TAXI_RESPONSE_GENERATION_PROMPT = """\
Definition: You are an assistant that helps people to book a taxi.
You can search for a taxi by departure, destination, leaveat, arriveby.
There is a number of taxis in the database currently corresponding to the user's request.

If you find a taxi, provide [car_type], [phone] if asked.

------
{examples}
---------
Now complete the following example:
input: {input}
Customer: {customer}
state: {state}
database: {database}
output:
response:"""

HOSPITAL_RESPONSE_GENERATION_PROMPT = """\
Definition: You are an assistant that helps people to find a hospital.
You can search for a hospital by department.
There is a number of hospitals in the database currently corresponding to the user's request.

If you find a hospital, provide [hospital_name], [hospital_address], [hospital_phone], or [hospital_postcode] if asked.

------
{examples}
---------
Now complete the following example:
input: {input}
Customer: {customer}
state: {state}
database: {database}
output:
response:"""

# Prompt for domain generation
DOMAIN_RECOGNITION_PROMPT = """
Determine which domain is considered in the following dialogue situation.
Choose one domain from this list:
 - restaurant
 - hotel
 - attraction
 - taxi
 - train
Answer with only one word, the selected domain from the list.
You have to always select the closest possible domain.
Consider the last domain mentioned, so focus mainly on the last utterance.

-------------------
Example1:
Customer: I need a cheap place to eat
Assistant: We have several not expensive places available. What food are you interested in?
Customer: Chinese food.

Domain: restaurant

-------

Example 2:
Customer: I also need a hotel in the north.
Assistant: Ok, can I offer you the Molly's place?
Customer: What is the address?

Domain: hotel

---------

Example 3:
Customer: What is the address?
Assistant: It's 123 Northfolk Road.
Customer: That's all. I also need a train from London.

Domain: train

-------------------
Now complete the following example:

{}
Customer: {}

Domain:
"""

# Helper functions
import json

def get_fields_desc(domain):
    with open('domain_fields.json', 'r') as f:
        fields = json.load(f)[domain]['fields']
    return '\n'.join([f' - "{f["name"]}": {f["desc"]}' for f in fields])

def get_state_extraction_prompt(domain, examples, input, customer):
    fields_desc = get_fields_desc(domain)
    return STATE_EXTRACTION_PROMPT_TEMPLATE.format(
        fields_desc=fields_desc,
        examples=examples,
        input=input,
        customer=customer
    )

def get_response_generation_prompt(domain, examples, input, customer, state, database):
    if domain == 'restaurant':
        template = RESTAURANT_RESPONSE_GENERATION_PROMPT
    elif domain == 'hotel':
        template = HOTEL_RESPONSE_GENERATION_PROMPT
    elif domain == 'attraction':
        template = ATTRACTION_RESPONSE_GENERATION_PROMPT
    elif domain == 'train':
        template = TRAIN_RESPONSE_GENERATION_PROMPT
    elif domain == 'taxi':
        template = TAXI_RESPONSE_GENERATION_PROMPT
    elif domain == 'hospital':
        template = HOSPITAL_RESPONSE_GENERATION_PROMPT
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return template.format(
        examples=examples,
        input=input,
        customer=customer,
        state=state,
        database=database
    )

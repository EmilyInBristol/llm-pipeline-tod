# prompts.py

# Prompt for state extraction
STATE_EXTRACTION_PROMPT = """\
Capture entity values from last utterance of the conversation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" with no spaces.
Separate entity:value pairs by hyphens.

Values that should be captured are:
 - "pricerange": price range of the restaurant (cheap/moderate/expensive)
 - "area": area where the restaurant is located (north/east/west/south/centre)
 - "food": type of food the restaurant serves
 - "name": name of the restaurant
 - "bookday": day of the booking
 - "booktime": time of the booking
 - "bookpeople": how many people the booking is for

Do not capture any other values!
If a value is not specified, leave it empty.

------
{}
---------
Now complete the following example:
input: {}
Customer: {}
output:
state:"""

# Prompt for response generation
RESPONSE_GENERATION_PROMPT = """\
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is a number of restaurants in the database currently corresponding to the user's request.

If you find a restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone], or [restaurant_postcode] if asked.
If booking, provide [reference] in the answer.
Always act as if the booking is successfully done.

------
{}{}
---------
Now complete the following example:
input: {}
Customer: {}
state: {}
database: {}
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

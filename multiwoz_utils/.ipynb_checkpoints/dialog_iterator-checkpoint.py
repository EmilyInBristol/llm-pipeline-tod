import json
from collections import defaultdict

DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']
TOTAL = 50

def iterate_dialogues(data, database, context_size=3, domains=DOMAINS, max_dialogs=TOTAL):
    """
    Generator that yields structured turn-level data from dialogues.
    
    Args:
        data: List of dialogue dicts.
        database: A database object with .query(domain, domain_state).
        context_size: How many turns to include as context (default: 3).
        domains: List of supported domains (default: MultiWOZ-style domains).
        max_dialogs: Max number of dialogues per domain to include.
    
    Yields:
        dict: Turn-level structured data including context, state, and metadata.
    """
    domain_counts = {d: 0 for d in domains}
    
    for dialog in data:
        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        domain_gt = dialog['services'][0] if dialog['services'] else ''

        if domain_gt not in domains or domain_counts[domain_gt] >= max_dialogs:
            continue
        domain_counts[domain_gt] += 1
        
        last_state = {}

        for tn in range(0, len(dialog['turns']['utterance']), 2):
            # Build context
            context = [
                f"Customer: {t}" if i % 2 == 0 else f"Assistant: {t}"
                for i, t in enumerate(dialog['turns']['utterance'][:tn+1])
            ]

            # Parse current turn state
            state = dialog['turns']['frames'][tn]['state']
            if not state:
                state = {}
            else:
                state = state[0]['slots_values']
                state = {
                    k: v[0]
                    for k, v in zip(state['slots_values_name'], state['slots_values_list'])
                }

            # Structure the current full state
            new_state = {}
            for sl, val in state.items():
                domain, name = sl.split('-')
                new_state.setdefault(domain, {})[name] = val

            # Compute state updates
            state_update = {}
            for domain, domain_state in new_state.items():
                for slot, value in domain_state.items():
                    if slot not in last_state.get(domain, {}) or last_state[domain][slot] != value:
                        state_update.setdefault(domain, {})[slot] = value

            last_state = new_state

            # Query the database
            database_results = {
                domain: len(database.query(domain, domain_state))
                for domain, domain_state in new_state.items()
            }

            yield {
                'page_content': '\n'.join(context[-context_size:]),
                'question': dialog['turns']['utterance'][tn],
                'gt_state': last_state,
                'dialogue_id': dialogue_id,
                'metadata': {
                    'domain': domain_gt,
                    'state': state_update,
                    'full_state': last_state,
                    'context': '\n'.join(context[-6:]),
                    'response': dialog['turns']['utterance'][tn + 1],
                    'database': database_results
                }
            }

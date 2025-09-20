import re

# --- US to AU Spelling Dictionary ---
# This list is not exhaustive but covers many common cases.
SPELLING_MAP = {
    # -ize -> -ise
    'organize': 'organise', 'organizes': 'organises', 'organizing': 'organising', 'organized': 'organised',
    'recognize': 'recognise', 'recognizes': 'recognises', 'recognizing': 'recognising', 'recognized': 'recognised',
    'realize': 'realise', 'realizes': 'realises', 'realizing': 'realising', 'realized': 'realised',
    'specialize': 'specialise', 'specializes': 'specialises', 'specializing': 'specialising', 'specialized': 'specialised',
    'apologize': 'apologise', 'apologizes': 'apologises', 'apologizing': 'apologising', 'apologized': 'apologised',
    'criticize': 'criticise', 'criticizes': 'criticises', 'criticizing': 'criticising', 'criticized': 'criticised',
    
    # -yze -> -yse
    'analyze': 'analyse', 'analyzes': 'analyses', 'analyzing': 'analysing', 'analyzed': 'analysed',
    'paralyze': 'paralyse', 'paralyzes': 'paralyses', 'paralyzing': 'paralysing', 'paralyzed': 'paralysed',

    # -or -> -our
    'color': 'colour', 'colors': 'colours',
    'flavor': 'flavour', 'flavors': 'flavours',
    'humor': 'humour', 'humors': 'humours',
    'labor': 'labour', 'labors': 'labours',
    'neighbor': 'neighbour', 'neighbors': 'neighbours',
    'honor': 'honour', 'honors': 'honours',
    'rumor': 'rumour', 'rumors': 'rumours',
    'splendor': 'splendour',
    'vapor': 'vapour',
    'endeavor': 'endeavour', 'endeavors': 'endeavours',

    # -er -> -re
    'center': 'centre', 'centers': 'centres',
    'theater': 'theatre', 'theaters': 'theatres',
    'meter': 'metre', 'meters': 'metres',
    'liter': 'litre', 'liters': 'litres',
    'fiber': 'fibre', 'fibers': 'fibres',
    'saber': 'sabre', 'sabers': 'sabres',

    # -ense -> -ence
    'defense': 'defence', 'defenses': 'defences',
    'offense': 'offence', 'offenses': 'offences',
    'pretense': 'pretence', 'pretenses': 'pretences',
    'license': 'licence', # Note: verb 'license' is same, but noun is 'licence' in AU. This is a simplification.

    # -og -> -ogue
    'catalog': 'catalogue', 'catalogs': 'catalogues',
    'dialog': 'dialogue', 'dialogs': 'dialogues',
    'analog': 'analogue', 'analogs': 'analogues',

    # -el variants
    'traveling': 'travelling', 'traveled': 'travelled', 'traveler': 'traveller',
    'modeling': 'modelling', 'modeled': 'modelled',
    'fueling': 'fuelling', 'fueled': 'fuelled',
    'labeling': 'labelling', 'labeled': 'labelled',
    'signaling': 'signalling', 'signaled': 'signalled',

    # Misc
    'gray': 'grey',
    'program': 'programme', 'programs': 'programmes',
    'airplane': 'aeroplane', 'airplanes': 'aeroplanes',
    'draft': 'draught',
    'skeptic': 'sceptic', 'skeptical': 'sceptical', 'skepticism': 'scepticism',
    'jewelry': 'jewellery',
    'maneuver': 'manoeuvre', 'maneuvers': 'manoeuvres',
    'plow': 'plough', 'plows': 'ploughs',
    'practice': 'practise', # verb
    'story': 'storey', # building level
}

# --- Exceptions for spelling conversion ---
# Words in this set will not be converted, regardless of SPELLING_MAP.
EXCEPTIONS = {
    "program", "programming", "programmer", "programs", "programmes", # Computing terms
    "parameter", "parameters",
    "thermometer",
    "analogue", # Already AU spelling, but might be in a context where it shouldn't be changed
    "dialogue", # Already AU spelling
    "catalogue", # Already AU spelling
    "disk", "disks", # Storage
    "byte", "bytes",
    "colour", "colours", # Already AU spelling
    "flavour", "flavours", # Already AU spelling
    "humour", "humours", # Already AU spelling
    "labour", "labours", # Already AU spelling
    "neighbour", "neighbours", # Already AU spelling
    "honour", "honours", # Already AU spelling
    "rumour", "rumours", # Already AU spelling
    "splendour",
    "vapour",
    "endeavour", "endeavours",
    "centre", "centres",
    "theatre", "theatres",
    "metre", "metres",
    "litre", "litres",
    "fibre", "fibres",
    "sabre", "sabres",
    "defence", "defences",
    "offence", "offences",
    "pretence", "pretences",
    "licence",
    "aeroplane", "aeroplanes",
    "draught",
    "sceptic", "sceptical", "scepticism",
    "jewellery",
    "manoeuvre", "manoeuvres",
    "plough", "ploughs",
    "practise",
    "storey",
}
# This list should be reviewed and expanded as needed for specific corpora to prevent unintended conversions.

def convert_text_to_au_english(text: str) -> str:
    """
    Converts US English spellings in the given text to Australian English spellings.
    Handles common spelling differences, including capitalization, while respecting an exceptions list.
    """
    converted_text = text
    for us_word, au_word in SPELLING_MAP.items():
        # Create a regex pattern for the word, ensuring it's a whole word
        pattern = re.compile(r'\b' + re.escape(us_word) + r'\b', re.IGNORECASE)
        
        def replace_func(match):
            original_word = match.group(0)
            # Check if the original word (case-insensitively) is in the exceptions list
            if original_word.lower() in EXCEPTIONS:
                return original_word # Do not convert if it's an exception

            # Handle capitalization: if the found word is capitalized, capitalize the replacement
            if original_word.istitle():
                return au_word.title()
            if original_word.isupper():
                return au_word.upper()
            return au_word

        converted_text = pattern.sub(replace_func, converted_text)
    return converted_text

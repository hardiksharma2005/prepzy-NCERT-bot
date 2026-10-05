"""Hand-curated word lists the cache guards rely on."""

# Words that carry no meaning for "which question is this?" once intent is extracted.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done", "has", "have", "had", "having",
    "what", "whats", "which", "who", "whom", "whose", "where", "when", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must",
    "of", "in", "on", "at", "by", "for", "with", "about", "as", "into", "onto",
    "from", "to", "towards", "toward", "through", "via", "per", "upon", "within",
    "and", "or", "but", "so", "if", "then", "than", "also", "too", "very", "just",
    "i", "me", "my", "we", "our", "you", "your", "u", "pls", "plz", "please", "kindly",
    "tell", "know", "want", "understand", "help", "doubt", "question", "answer",
    "some", "any", "all", "each", "every", "such", "there", "here", "its", "it",
    "this", "that", "these", "those", "they", "them", "their", "he", "she",
    "sir", "maam", "mam", "hey", "hi", "hello", "ok", "okay",
    "actually", "exactly", "basically", "really", "briefly", "brief", "simple",
    "detail", "detailed", "properly", "clearly", "ncert", "class",
    "textbook", "book", "chapter", "science",
    # intent words: captured separately by `detect_intent`, so dropped from terms
    "define", "definition", "mean", "meaning", "meant", "called", "term", "terms",
    "explain", "explanation", "describe", "description", "elaborate", "discuss",
    "why", "reason", "reasons", "how", "differ", "difference", "differences", "differentiate",
    "distinguish", "between", "compare", "comparison", "versus", "vs", "contrast",
    "example", "examples", "instance", "instances", "give", "name", "list", "state",
    "write", "mention", "identify", "type", "types", "kind", "kinds",
    "use", "uses", "application", "applications", "function", "functions", "role",
    "calculate", "find", "compute", "determine", "value", "happen",
    "happens", "occur", "occurs", "process", "phenomenon", "concept", "meaning",
    "appear", "appears", "seem", "seems",
    "draw", "diagram", "labelled", "labeled", "sketch", "show",
}

# Words that, if present in one question but not the other, do not change the question.
# Kept deliberately tiny: every word here is a potential false hit.
GENERIC_TERMS = {"light", "word", "thing", "things", "statement", "formula", "topic"}

# British/American and common shorthand spellings -> one canonical form.
SYNONYMS = {
    "color": "colour", "colors": "colour", "colours": "colour",
    "fiber": "fibre", "center": "centre", "meter": "metre", "liter": "litre",
    "sulfur": "sulphur", "sulfuric": "sulphuric", "sulfate": "sulphate",
    "sulfide": "sulphide", "sulfurous": "sulphurous",
    "aluminum": "aluminium", "vapor": "vapour", "hemoglobin": "haemoglobin",
    "esophagus": "oesophagus", "fetus": "foetus", "estrogen": "oestrogen",
    "behavior": "behaviour", "analyze": "analyse", "ionized": "ionised",
    "eqn": "equation", "rxn": "reaction", "reac": "reaction",
    "h2o": "water", "co2": "carbon-dioxide",
    "ohms": "ohm", "volts": "volt", "amperes": "ampere", "amps": "ampere", "amp": "ampere",
    "watts": "watt", "joules": "joule", "dioptres": "dioptre", "diopter": "dioptre",
    "diopters": "dioptre", "cms": "cm",
    "ac": "alternating", "dc": "direct",
}

# Pairs that look alike to an embedding model but are different questions.
# Term-set equality already separates most of these; this is an explicit second check.
CONTRAST_PAIRS = [
    ("concave", "convex"), ("converging", "diverging"), ("real", "virtual"),
    ("erect", "inverted"), ("magnified", "diminished"), ("enlarged", "diminished"),
    ("mirror", "lens"), ("reflection", "refraction"), ("rarer", "denser"),
    ("myopia", "hypermetropia"), ("myopia", "presbyopia"), ("hypermetropia", "presbyopia"),
    ("acid", "base"), ("acidic", "basic"), ("acid", "alkali"), ("strong", "weak"),
    ("metal", "non-metal"), ("metallic", "non-metallic"), ("ionic", "covalent"),
    ("oxidation", "reduction"), ("oxidised", "reduced"), ("exothermic", "endothermic"),
    ("combination", "decomposition"), ("displacement", "double-displacement"),
    ("saturated", "unsaturated"), ("alkane", "alkene"), ("alkene", "alkyne"),
    ("aerobic", "anaerobic"), ("autotrophic", "heterotrophic"), ("xylem", "phloem"),
    ("artery", "vein"), ("arteries", "veins"), ("inhalation", "exhalation"),
    ("sexual", "asexual"), ("male", "female"), ("dominant", "recessive"),
    ("tall", "short"), ("plant", "animal"), ("herbivore", "carnivore"),
    ("biodegradable", "non-biodegradable"), ("producer", "consumer"),
    ("voluntary", "involuntary"), ("sensory", "motor"), ("auxin", "gibberellin"),
    ("series", "parallel"), ("alternating", "direct"), ("ac", "dc"),
    ("resistance", "resistivity"), ("current", "voltage"), ("potential", "current"),
    ("north", "south"), ("clockwise", "anticlockwise"), ("motor", "generator"),
    ("increase", "decrease"), ("increases", "decreases"), ("more", "less"),
    ("maximum", "minimum"), ("higher", "lower"), ("advantage", "disadvantage"),
    ("advantages", "disadvantages"), ("before", "after"), ("first", "second"),
]

NEGATIONS = {"not", "no", "never", "without", "except", "cannot", "cant", "dont",
             "doesnt", "isnt", "arent", "wont", "neither", "nor", "none", "unable"}

# Short wordings that need the conversation to make sense: never cached as-is.
PRONOUNS = {"it", "its", "they", "them", "their", "theirs", "these", "those",
            "he", "she", "him", "her", "his", "itself", "themselves"}
DEMONSTRATIVES = {"this", "that"}

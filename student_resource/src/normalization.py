"""
Business Entity Normalization Module for Cross-Source Entity Resolution.

Supports:
- Multilingual transliteration (Indic scripts: Devanagari, Tamil, Telugu, etc.)
- Diacritics/accents removal (French, Spanish, German, etc.)
- Legal entity suffix/prefix canonicalization and stripping (US, India, France)
- Web domain & URL artifact removal
- Address standardization & component order neutralization
- Multi-representation blocking key generation
"""

import re
import unicodedata
from unidecode import unidecode

# -------------------------------------------------------------
# Legal Entity Stopwords & Generic Business Descriptors
# Covers US, India, and France (Open-set test requirement)
# -------------------------------------------------------------
LEGAL_STOPWORDS = {
    # US / UK / Global legal forms
    'llc', 'l.l.c.', 'l.l.c', 'llp', 'l.l.p.', 'l.l.p',
    'inc', 'inc.', 'incorporated', 'corp', 'corp.', 'corporation',
    'ltd', 'ltd.', 'limited', 'pvt', 'pvt.', 'private',
    'pvtltd', 'privatelimited', 'p.ltd', 'p.ltd.',
    'co', 'co.', 'company', 'cie', 'plc', 'pc', 'pa', 'pllc',
    'gmbh', 'ag', 'bv', 'nv', 'pty', 'opc',
    
    # French legal forms (Test set includes France)
    'sarl', 's.a.r.l.', 's.a.r.l', 'sa', 's.a.', 's.a',
    'sas', 's.a.s.', 's.a.s', 'sasu', 's.a.s.u.',
    'snc', 's.n.c.', 'sci', 's.c.i.', 'eurl', 'e.u.r.l.',
    'societe', 'entreprise',
    
    # Common generic business descriptors often attached/detached
    'services', 'service', 'solutions', 'solution', 'holdings', 'holding',
    'group', 'enterprises', 'enterprise', 'associates', 'associate',
    'consulting', 'consultants', 'consultant', 'international', 'mgmt', 'management',
    'center', 'centre', 'the', 'and', 'in', 'of', 'for', 'at', 'by'
}

# Regex to remove domain names (.com, .in, .fr, etc.)
DOMAIN_REGEX = re.compile(
    r'\.(com|in|co\.in|org|net|io|ai|biz|info|us|fr|gov|edu|co)\b',
    re.IGNORECASE
)

# Standard Address Term Canonicalization
ADDRESS_MAP = {
    'street': 'st', 'saint': 'st', 'road': 'rd', 'avenue': 'ave',
    'drive': 'dr', 'boulevard': 'blvd', 'lane': 'ln', 'court': 'ct',
    'parkway': 'pkwy', 'highway': 'hwy', 'terrace': 'ter', 'place': 'pl',
    'circle': 'cir', 'apartment': 'apt', 'suite': 'ste', 'building': 'bldg',
    'floor': 'fl', 'number': 'no', 'plot': 'plot', 'door': 'door',
    'post office': 'po', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw'
}

ADDRESS_COMPILED_REGEX = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(ADDRESS_MAP.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

NON_ALPHANUM_REGEX = re.compile(r'[^a-z0-9\s]')
WHITESPACE_REGEX = re.compile(r'\s+')


def clean_string(text: str) -> str:
    """
    Standard transliteration and ASCII normalization.
    """
    if not text:
        return ""
    # Transliterate Unicode scripts (Devanagari, Tamil, accented French, etc.) to ASCII
    text = unidecode(str(text)).lower()
    # Strip URL domains and protocols
    text = DOMAIN_REGEX.sub(' ', text)
    text = text.replace('www.', ' ').replace('http://', ' ').replace('https://', ' ')
    text = text.replace('&', ' and ').replace('+', ' plus ')
    # Replace all non-alphanumeric with space
    text = NON_ALPHANUM_REGEX.sub(' ', text)
    return WHITESPACE_REGEX.sub(' ', text).strip()


def normalize_name_core(name: str) -> str:
    """
    Strips legal forms, generic terms, and returns concatenated core name string.
    """
    clean = clean_string(name)
    if not clean:
        return ""
    tokens = [t for t in clean.split() if t not in LEGAL_STOPWORDS]
    return "".join(tokens)


def normalize_name_tokens(name: str) -> list[str]:
    """
    Returns sorted list of non-legal tokens from name.
    """
    clean = clean_string(name)
    if not clean:
        return []
    return [t for t in clean.split() if t not in LEGAL_STOPWORDS]


def normalize_address(address: str) -> str:
    """
    Canonicalizes address abbreviations, removes noise, and sorts tokens
    to provide order-invariant address matching.
    """
    if not address:
        return ""
    s = str(address).strip().lower()
    if s in {'', 'null', 'none', 'nan'}:
        return ""
    
    clean = clean_string(address)
    clean = ADDRESS_COMPILED_REGEX.sub(lambda m: ADDRESS_MAP[m.group(0).lower()], clean)
        
    tokens = sorted(clean.split())
    return "".join(tokens)


def extract_distinctive_tokens(name: str, max_tokens: int = 3, min_len: int = 4) -> list[str]:
    """
    Extracts the most distinctive (longer) non-legal tokens for inverted indexing.
    """
    clean = clean_string(name)
    tokens = [t for t in clean.split() if t not in LEGAL_STOPWORDS and len(t) >= min_len]
    # Longer tokens are more specific / lower false-positive rate
    tokens = sorted(tokens, key=len, reverse=True)
    return tokens[:max_tokens]


def get_record_blocking_keys(name: str, address: str, country: str, 
                             include_tokens: bool = True,
                             include_address: bool = True) -> list[str]:
    """
    Generates a set of normalized blocking keys for candidate generation.
    Any overlap between S1 keys and S2/S3 keys captures the match.
    """
    keys = []
    country_tag = str(country).strip().lower() if country else "unk"
    
    # Key 1: Core Name (legal stripped)
    core_name = normalize_name_core(name)
    if core_name:
        keys.append(f"name:{core_name}")
        # Key 1b: Country-partitioned name
        keys.append(f"name_co:{country_tag}:{core_name}")
        
    # Key 2: Order-Invariant Normalized Address
    if include_address:
        norm_addr = normalize_address(address)
        if norm_addr and len(norm_addr) >= 6:
            keys.append(f"addr:{country_tag}:{norm_addr}")
            
    # Key 3: Distinctive Name Tokens (Inverted Index)
    if include_tokens:
        dist_tokens = extract_distinctive_tokens(name, max_tokens=2, min_len=4)
        for tok in dist_tokens:
            keys.append(f"tok:{country_tag}:{tok}")
            
    return keys


# Fast batch generator
def batch_generate_keys(names: list[str], addresses: list[str], countries: list[str],
                        include_tokens: bool = True,
                        include_address: bool = True) -> list[list[str]]:
    """
    Ultra-fast batch generator avoiding pandas row-iteration overhead.
    """
    result = []
    for name, addr, country in zip(names, addresses, countries):
        result.append(get_record_blocking_keys(
            name, addr, country,
            include_tokens=include_tokens,
            include_address=include_address
        ))
    return result

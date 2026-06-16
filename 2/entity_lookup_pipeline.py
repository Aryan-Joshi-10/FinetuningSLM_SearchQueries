"""
entity_lookup_pipeline.py

Standalone reference implementation of the rule-based entity extraction
pipeline used to auto-annotate the JioTV search-query dataset
(annotations.jsonl - 5k, jiotv_augmented_3000_annotations.jsonl - 3k).

Use this module for POST-PROCESSING the output of a fine-tuned model:
  - Validate / correct entity_name spellings against the canonical lookup
  - Snap a fine-tuned model's raw keyword span to a canonical entity_name
  - Re-run on queries the model gets wrong, as a fallback / ensemble member
  - Detect out-of-vocabulary entities the model may hallucinate

================================================================================
RULES SUMMARY
================================================================================

1. ENTITY TYPES (closed set, from the product spec screenshot):
     - language       (Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi,
                        Bengali, Punjabi, Gujarati, Bhojpuri)
     - content_type    (movies, shows, live_channels, music_videos, videos, trailers)
     - genre           (Comedy, Drama, Action, Thriller, Horror, Romance, ...)
     - starcast        (actor names - canonicalized from nicknames/aliases)
     - director        (canonicalized from nicknames/aliases)
     - singer          (not populated in this dataset - vocabulary placeholder)
     - musicDirector   (not populated in this dataset - vocabulary placeholder)
     - title           (movie/show titles - canonicalized, typo-tolerant)

2. EVERY token in the query is checked against the dictionaries below.
   Matching priority (most-specific first), each match "consumes" its
   token span so nothing is double-tagged:
     title > starcast > director > language > industry_slang(=language)
           > genre > content_type

3. USER-INTENT PHRASES are NEVER emitted as entities. These carry no
   entity_type/entity_name and are dropped from Output entirely:
     recommend, suggest karo, batao, dekhna, best of, review, rating,
     available hai kya, where to watch, kahan dekhein, worth watching,
     i want to watch, top 10, ever made, jaise aur koi hai kya,
     jaisi ... suggest karo, movies like / similar to / if you liked,
     movies by / directed by (when not asking for `director` explicitly),
     best, top, underrated, overrated, classic, award winning, latest, new,
     wali / wala (possessive/filler), dikhao, download, list.

4. EXACT MATCH FIRST, then FUZZY MATCH for typo-tolerance (only used on the
   3k augmented/typo dataset, see FUZZY LOGIC below). Exact dictionary
   lookups should ALWAYS be tried first in post-processing.

5. CANONICALIZATION: entity_name is always the canonical form (e.g.
   "akki" -> "Akshay Kumar", "bahubali"/"shahenshah" -> "Baahubali"),
   while `keyword` preserves the literal substring/typo from the query
   for span-alignment purposes.

================================================================================
FUZZY LOGIC (used only for typo-heavy augmented data)
================================================================================

  - Tokenize query into words (regex: [a-z0-9']+(?:-[a-z0-9']+)?), lowercased.
  - Pre-split tokens like "girl2" -> "girl","2" (alpha+digit glued together).
  - For n-gram sizes from MAXLEN down to 1 (MAXLEN = longest dictionary key
    in words, e.g. "dilwale dulhania le jayenge" = 5):
      a. Try EXACT match of the n-gram phrase against dictionary keys of
         that same word-length.
      b. If no exact match, and the phrase is >= FUZZY_MIN_LEN (4) chars
         and NOT in FUZZY_STOPLIST, try difflib.get_close_matches with
         cutoff = FUZZY_CUTOFF (0.78). Reject the match if
         abs(len(matched_key) - len(phrase)) > 2 (prevents swallowing
         neighboring words).
      c. If still no match, try comparing the phrase with spaces removed
         against dictionary keys with spaces removed (handles typos that
         merge/split words across a space, e.g. "om g2" -> "omg2" ->
         "omg 2"). Accept only if length difference <= 1.
  - FUZZY_STOPLIST: common structural/intent words that must NEVER be
    fuzzy-matched even at high similarity (critical fix: "best" has 0.89
    similarity to "Beast" and would otherwise be mis-tagged as the title
    "Beast" in almost every "best ... movies like X" query).

================================================================================
"""

import re
import difflib

# ---------------------------------------------------------------------------
# 1. LANGUAGE
# ---------------------------------------------------------------------------
LANGUAGES = {
    'hindi': 'Hindi', 'tamil': 'Tamil', 'telugu': 'Telugu', 'malayalam': 'Malayalam',
    'kannada': 'Kannada', 'marathi': 'Marathi', 'bengali': 'Bengali', 'bangla': 'Bengali',
    'punjabi': 'Punjabi', 'gujarati': 'Gujarati', 'bhojpuri': 'Bhojpuri',
    'telugu lo': 'Telugu',
}

# Regional-industry slang that implies a language
INDUSTRY_SLANG = {
    'bollywood': 'Hindi', "b'wood": 'Hindi',
    'tollywood': 'Telugu', "t'wood": 'Telugu',
    'kollywood': 'Tamil', "k'wood": 'Tamil',
    'mollywood': 'Malayalam', 'mallu': 'Malayalam',
    'sandalwood': 'Kannada',
}

# ---------------------------------------------------------------------------
# 2. GENRE
# ---------------------------------------------------------------------------
GENRES = {
    'comedy': 'Comedy', 'dark comedy': 'Dark Comedy', 'thriller': 'Thriller',
    'psychological thriller': 'Psychological Thriller', 'action': 'Action',
    'drama': 'Drama', 'romance': 'Romance', 'romantic': 'Romance',
    'horror': 'Horror', 'scary': 'Horror', 'mystery': 'Mystery',
    'crime': 'Crime', 'biopic': 'Biopic', 'historical': 'Historical',
    'sports': 'Sports', 'supernatural': 'Supernatural', 'sci-fi': 'Sci-Fi',
    'sci fi': 'Sci-Fi', 'family': 'Family', 'inspirational': 'Inspirational',
    'motivational': 'Inspirational', 'emotional': 'Emotional', 'sad': 'Emotional',
    'tearjerker': 'Emotional', 'suspenseful': 'Thriller', 'intense': 'Thriller',
    'funny': 'Comedy', 'light hearted': 'Comedy', 'light-hearted': 'Comedy',
    'feel good': 'Family',
}

# ---------------------------------------------------------------------------
# 3. CONTENT TYPE  (closed set per product spec)
# ---------------------------------------------------------------------------
CONTENT_TYPES = {
    'movies': 'movies', 'movie': 'movies', 'film': 'movies', 'films': 'movies',
    'full movie': 'movies', 'blockbusters': 'movies', 'blockbuster': 'movies',
    'shows': 'shows', 'show': 'shows', 'series': 'shows', 'web series': 'shows',
    'web-series': 'shows', 'original series': 'shows', 'original': 'shows',
    'binge worthy': 'shows', 'trailers': 'trailers', 'trailer': 'trailers',
    'music videos': 'music_videos', 'music video': 'music_videos',
    'videos': 'videos', 'video': 'videos',
    'live channels': 'live_channels', 'live channel': 'live_channels',
}

# ---------------------------------------------------------------------------
# 4. STARCAST  (nickname / alias -> canonical name)
# ---------------------------------------------------------------------------
STARCAST = {
    'vijay': 'Vijay', 'taapsee': 'Taapsee Pannu', 'bebo': 'Kareena Kapoor Khan',
    'kareena': 'Kareena Kapoor Khan', 'kangana': 'Kangana Ranaut',
    'ranbir': 'Ranbir Kapoor', 'ranveer': 'Ranveer Singh', 'ranveer singh': 'Ranveer Singh',
    'akshay': 'Akshay Kumar', 'akki': 'Akshay Kumar', 'khiladi': 'Akshay Kumar',
    'salman': 'Salman Khan', 'sallu': 'Salman Khan', 'bhai': 'Salman Khan',
    'shahrukh': 'Shah Rukh Khan', 'shah rukh': 'Shah Rukh Khan', 'srk': 'Shah Rukh Khan',
    'badshah': 'Shah Rukh Khan', 'king khan': 'Shah Rukh Khan', 'shah': 'Shah Rukh Khan',
    'aamir': 'Aamir Khan', 'mr perfectionist': 'Aamir Khan',
    'fahadh': 'Fahadh Faasil', 'fahad': 'Fahadh Faasil',
    'vidya': 'Vidya Balan', 'vidya balan': 'Vidya Balan',
    'alia': 'Alia Bhatt', 'alia bhatt': 'Alia Bhatt',
    'vicky kaushal': 'Vicky Kaushal', 'vicky': 'Vicky Kaushal',
    'tiger shroff': 'Tiger Shroff', 'tiger': 'Tiger Shroff',
    'kartik': 'Kartik Aaryan',
    'prabhas': 'Prabhas', 'rebel star': 'Prabhas', 'rebel star prabhas': 'Prabhas',
    'darling': 'Prabhas',
    'vijay deverakonda': 'Vijay Deverakonda', 'arjun reddy wala': 'Vijay Deverakonda',
    'mohanlal': 'Mohanlal', 'lalettan': 'Mohanlal',
    'mammootty': 'Mammootty', 'mammukka': 'Mammootty',
    'dulquer': 'Dulquer Salmaan', 'dq': 'Dulquer Salmaan', 'greek god': 'Dulquer Salmaan',
    'kamal': 'Kamal Haasan', 'ulaganayagan': 'Kamal Haasan',
    'allu': 'Allu Arjun', 'allu arjun': 'Allu Arjun', 'bunny': 'Allu Arjun',
    'rajinikanth': 'Rajinikanth', 'rajini': 'Rajinikanth', 'thalaivar': 'Rajinikanth',
    'superstar': 'Rajinikanth',
    'amitabh': 'Amitabh Bachchan', 'amitabh bachchan': 'Amitabh Bachchan',
    'bachchan sahab': 'Amitabh Bachchan', 'big b': 'Amitabh Bachchan',
    'irfan khan': 'Irrfan Khan', 'irrfan': 'Irrfan Khan', 'irfan': 'Irrfan Khan',
    'nawazuddin': 'Nawazuddin Siddiqui', 'nawaz': 'Nawazuddin Siddiqui',
    'katrina': 'Katrina Kaif', 'kat': 'Katrina Kaif',
    'priyanka': 'Priyanka Chopra', 'piggy chops': 'Priyanka Chopra', 'pc': 'Priyanka Chopra',
    'deepika': 'Deepika Padukone',
    'hrithik': 'Hrithik Roshan',
    'ayushmann': 'Ayushmann Khurrana', 'ayushman': 'Ayushmann Khurrana',
    'vijay deverakonda': 'Vijay Deverakonda',
    'rk': 'Ranbir Kapoor',
    'rocky': 'Ranbir Kapoor',
    'duggu': 'Allu Arjun',
}

# ---------------------------------------------------------------------------
# 5. DIRECTOR  (nickname / alias -> canonical name)
# ---------------------------------------------------------------------------
DIRECTORS = {
    'mani ratnam': 'Mani Ratnam',
    'imtiaz ali': 'Imtiaz Ali',
    'ss rajamouli': 'S. S. Rajamouli', 'rajamouli': 'S. S. Rajamouli',
    'lokesh kanagaraj': 'Lokesh Kanagaraj', 'lokesh': 'Lokesh Kanagaraj',
    'anurag kashyap': 'Anurag Kashyap', 'kashyap': 'Anurag Kashyap',
    'karan johar': 'Karan Johar', 'kjo': 'Karan Johar',
    'rajkumar hirani': 'Rajkumar Hirani', 'hirani': 'Rajkumar Hirani',
    'vishal bhardwaj': 'Vishal Bhardwaj',
    'vetrimaaran': 'Vetrimaaran',
    'farhan akhtar': 'Farhan Akhtar', 'akhtar': 'Farhan Akhtar',
    'sanjay leela bhansali': 'Sanjay Leela Bhansali', 'bhansali': 'Sanjay Leela Bhansali',
    'zoya akhtar': 'Zoya Akhtar',
}

# ---------------------------------------------------------------------------
# 6. SINGER / MUSIC DIRECTOR
#    Not populated -- no queries in either dataset reference these entity
#    types. Kept here as empty dicts so the schema/pipeline stays complete
#    and future data can extend it without code changes elsewhere.
# ---------------------------------------------------------------------------
SINGERS = {}
MUSIC_DIRECTORS = {}

# ---------------------------------------------------------------------------
# 7. TITLE  (movie/show titles seen in the dataset -> canonical display name)
# ---------------------------------------------------------------------------
TITLES = {
    'scam 1992': 'Scam 1992', 'jawan': 'Jawan', 'kabir singh': 'Kabir Singh',
    'drishyam 2': 'Drishyam 2', 'drishyam': 'Drishyam', 'kartik calling kartik': 'Kartik Calling Kartik',
    'kgf chapter 2': 'KGF: Chapter 2', 'kgf': 'KGF: Chapter 2',
    'bahubali': 'Baahubali', 'baahubali': 'Baahubali', 'shahenshah': 'Baahubali',
    'rrr': 'RRR', 'haseen dillruba': 'Haseen Dillruba', 'kesari': 'Kesari',
    'roohi': 'Roohi', 'vikram vedha': 'Vikram Vedha', 'liger': 'Liger',
    'dunki': 'Dunki', 'pushpa': 'Pushpa', 'dobaaraa': 'Dobaaraa',
    'kushi': 'Kushi', 'pathaan': 'Pathaan', 'panchayat': 'Panchayat',
    'leo': 'Leo', 'tiger zinda hai': 'Tiger Zinda Hai', 'raees': 'Raees',
    'animal': 'Animal', 'an action hero': 'An Action Hero',
    'dream girl 2': 'Dream Girl 2', 'darlings': 'Darlings',
    'skanda': 'Skanda', 'brahmastra': 'Brahmastra', 'phone bhoot': 'Phone Bhoot',
    'dilwale dulhania le jayenge': 'Dilwale Dulhania Le Jayenge', 'ddlj': 'Dilwale Dulhania Le Jayenge',
    'kantara': 'Kantara', 'dabangg': 'Dabangg', 'omg 2': 'OMG 2',
    'gulabo sitabo': 'Gulabo Sitabo', 'jailer': 'Jailer', 'sultan': 'Sultan',
    'bajrangi bhaijaan': 'Bajrangi Bhaijaan', 'bard of blood': 'Bard of Blood',
    'manjummel boys': 'Manjummel Boys', '777 charlie': '777 Charlie',
    'kota factory': 'Kota Factory', 'mirzapur': 'Mirzapur', 'fukrey': 'Fukrey',
    'ludo': 'Ludo', 'thunivu': 'Thunivu', 'jalsa': 'Jalsa', 'varisu': 'Varisu',
    'sardar udham': 'Sardar Udham', 'gehraiyaan': 'Gehraiyaan', 'war': 'War',
    'mard ko dard nahi hota': 'Mard Ko Dard Nahi Hota', 'shershaah': 'Shershaah',
    'aspirants': 'Aspirants', 'sacred games': 'Sacred Games', 'uri': 'Uri: The Surgical Strike',
    'uri the surgical strike': 'Uri: The Surgical Strike',
    'andhadhun': 'AndhaDhun', 'bhool bhulaiyaa': 'Bhool Bhulaiyaa',
    'badhaai ho': 'Badhaai Ho', 'thappad': 'Thappad', 'gadar 2': 'Gadar 2',
    'gadar': 'Gadar 2', 'vikram': 'Vikram', 'pagglait': 'Pagglait',
    'article 15': 'Article 15', 'delhi crime': 'Delhi Crime',
    'beast': 'Beast', 'garuda gamana vrishabha vahana': 'Garuda Gamana Vrishabha Vahana',
    'ponniyin selvan': 'Ponniyin Selvan', 'hi nanna': 'Hi Nanna',
    'aadujeevitham': 'Aadujeevitham', 'bramayugam': 'Bramayugam',
    'dasara': 'Dasara', 'freddy': 'Freddy',
    'pk': 'PK', 'premalu': 'Premalu', '12th fail': '12th Fail', 'dangal': 'Dangal',
}

# ---------------------------------------------------------------------------
# 8. INTENT PHRASES (never emitted as entities - reference list only)
# ---------------------------------------------------------------------------
INTENT_PHRASES = [
    'recommend', 'suggest karo', 'recommend karo', 'batao', 'dekhna', 'best of',
    'review', 'rating', 'available hai kya', 'available on', 'is available',
    'kahan dekhein', 'worth watching', 'where to watch', 'kaha dekhe',
    'i want to watch', 'something', 'top 10', 'ever made', 'mein kya hai',
    'pe kya dekhein', 'ka rating kya hai', 'ki rating kya hai',
    'jaise aur koi hai kya', 'jesi film suggest karo', 'jaisi movie',
    'type movie', 'movies like', 'similar to', 'if you liked', 'movies by',
    'directed movies', 'wala', 'wali', 'ke saare movies', 'ki saare movies',
    'all movies list', 'best', 'top', 'underrated', 'overrated', 'classic',
    'award winning', 'latest', 'new', 'top rated', 'good', 'list',
    'dikhao', 'download',
]

# ---------------------------------------------------------------------------
# ENTITY-TYPE SEARCH ORDER (most-specific first)
# ---------------------------------------------------------------------------
ALL_DICTS = [
    (TITLES, 'title'),
    (STARCAST, 'starcast'),
    (DIRECTORS, 'director'),
    (LANGUAGES, 'language'),
    (INDUSTRY_SLANG, 'language'),
    (GENRES, 'genre'),
    (CONTENT_TYPES, 'content_type'),
    (SINGERS, 'singer'),
    (MUSIC_DIRECTORS, 'musicDirector'),
]

FLAT = []
for d, etype in ALL_DICTS:
    for k, v in d.items():
        FLAT.append((k, etype, v, len(k.split())))

MAXLEN = max((f[3] for f in FLAT), default=1)

KEYS_BY_NGRAM = {}
for k, etype, v, nw in FLAT:
    KEYS_BY_NGRAM.setdefault(nw, []).append((k, etype, v))

# ---------------------------------------------------------------------------
# FUZZY MATCH CONFIG
# ---------------------------------------------------------------------------
FUZZY_MIN_LEN = 4
FUZZY_CUTOFF = 0.78

FUZZY_STOPLIST = {
    'best', 'new', 'top', 'good', 'list', 'movie', 'movies', 'show', 'shows',
    'film', 'films', 'with', 'wali', 'wala', 'jaisi', 'jesi', 'jaise', 'like',
    'likes', 'liked', 'similar', 'starring', 'star', 'review', 'reviews',
    'rating', 'ratings', 'download', 'downloads', 'dikhao', 'batao', 'recommend',
    'suggest', 'series', 'video', 'videos', 'trailer', 'trailers', 'available',
    'watch', 'watching', 'worth', 'kahan', 'dekhein', 'dekhna', 'kaisa', 'kaise',
    'better', 'great', 'nice', 'okay', 'fine', 'recent', 'latest', 'classic',
    'underrated', 'overrated', 'award', 'winning', 'made', 'ever', 'type',
}


def normalize(q: str) -> str:
    return q.lower().strip()


def find_entities(query: str, use_fuzzy: bool = True):
    """
    Run the full rule-based pipeline on a single query string.

    Returns a list of dicts: {"keyword": ..., "entity_type": ..., "entity_name": ...}
    in left-to-right order, matching the format used in annotations.jsonl /
    jiotv_augmented_3000_annotations.jsonl.

    Set use_fuzzy=False to use ONLY exact dictionary lookups (recommended
    for clean, non-typo queries - e.g. validating model output against the
    canonical lookup table without risking fuzzy false positives).
    """
    qn = normalize(query)

    word_tokens = []
    for m in re.finditer(r"[a-z0-9']+(?:-[a-z0-9']+)?", qn):
        tok = m.group()
        start, end = m.start(), m.end()
        sub = re.match(r"^([a-z']+)(\d+)$", tok)
        if sub:
            word_tokens.append((sub.group(1), start, start + len(sub.group(1))))
            word_tokens.append((sub.group(2), start + len(sub.group(1)), end))
        else:
            word_tokens.append((tok, start, end))

    found = []
    used_token_idx = set()

    for n in range(MAXLEN, 0, -1):
        if n not in KEYS_BY_NGRAM:
            continue
        candidates = KEYS_BY_NGRAM[n]
        cand_keys = [c[0] for c in candidates]
        for i in range(len(word_tokens) - n + 1):
            idxs = list(range(i, i + n))
            if any(j in used_token_idx for j in idxs):
                continue
            phrase = ' '.join(word_tokens[j][0] for j in idxs)
            phrase_nospace = phrase.replace(' ', '')

            match_key = None
            if phrase in cand_keys:
                match_key = phrase
            elif use_fuzzy:
                if len(phrase) >= FUZZY_MIN_LEN and phrase not in FUZZY_STOPLIST:
                    close = difflib.get_close_matches(phrase, cand_keys, n=1, cutoff=FUZZY_CUTOFF)
                    if close and abs(len(close[0]) - len(phrase)) <= 2:
                        match_key = close[0]
                    else:
                        cand_keys_nospace = [c.replace(' ', '') for c in cand_keys]
                        close2 = difflib.get_close_matches(phrase_nospace, cand_keys_nospace, n=1, cutoff=FUZZY_CUTOFF)
                        if close2:
                            matched_nospace = close2[0]
                            if abs(len(matched_nospace) - len(phrase_nospace)) <= 1:
                                match_key = cand_keys[cand_keys_nospace.index(matched_nospace)]

            if match_key:
                for k, etype, v in candidates:
                    if k == match_key:
                        start = word_tokens[idxs[0]][1]
                        end = word_tokens[idxs[-1]][2]
                        found.append({
                            'start': start,
                            'keyword': query[start:end],
                            'entity_type': etype,
                            'entity_name': v,
                        })
                        for j in idxs:
                            used_token_idx.add(j)
                        break

    found.sort(key=lambda x: x['start'])
    for f in found:
        del f['start']
    return found


def annotate(query: str, use_fuzzy: bool = True) -> dict:
    """Convenience wrapper returning the {"search_query":..., "Output":...} shape."""
    return {"search_query": query, "Output": find_entities(query, use_fuzzy=use_fuzzy)}


# ---------------------------------------------------------------------------
# POST-PROCESSING HELPERS for fine-tuned model output
# ---------------------------------------------------------------------------

def canonical_entity_name(entity_type: str, raw_name: str, use_fuzzy: bool = True):
    """
    Given an entity_type and a (possibly mis-spelled / non-canonical) entity_name
    predicted by a fine-tuned model, return the canonical entity_name from the
    lookup table, or None if not found.

    Example:
        canonical_entity_name("starcast", "akki")        -> "Akshay Kumar"
        canonical_entity_name("title", "brahmastar")     -> "Brahmastra"  (fuzzy)
        canonical_entity_name("genre", "Sci Fi")         -> "Sci-Fi"
    """
    d = {
        'title': TITLES, 'starcast': STARCAST, 'director': DIRECTORS,
        'language': {**LANGUAGES, **INDUSTRY_SLANG}, 'genre': GENRES,
        'content_type': CONTENT_TYPES, 'singer': SINGERS, 'musicDirector': MUSIC_DIRECTORS,
    }.get(entity_type)
    if d is None:
        return None

    key = raw_name.lower().strip()
    if key in d:
        return d[key]

    # already-canonical form? (case-insensitive match on values)
    for v in d.values():
        if v.lower() == key:
            return v

    if use_fuzzy:
        keys = list(d.keys()) + list(d.values())
        close = difflib.get_close_matches(key, [k.lower() for k in keys], n=1, cutoff=FUZZY_CUTOFF)
        if close:
            match = close[0]
            for k, v in d.items():
                if k.lower() == match or v.lower() == match:
                    return v
    return None


def is_known_entity_type(entity_type: str) -> bool:
    return entity_type in {
        'title', 'starcast', 'director', 'language', 'genre',
        'content_type', 'singer', 'musicDirector',
    }


def is_valid_content_type_value(value: str) -> bool:
    """content_type entity_name must be one of the 6 closed-set values."""
    return value in {'movies', 'shows', 'live_channels', 'music_videos', 'videos', 'trailers'}


if __name__ == '__main__':
    # quick smoke test
    tests = [
        "vijay film scam 1992",
        "marathi comedy web series",
        "12t hfail review",
        "best dark comedy movies like Beast",
        "barhmastra review",
    ]
    for t in tests:
        print(annotate(t))

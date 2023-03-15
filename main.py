import fitz
import spacy
import pandas as pd
import re
from spacy.matcher import Matcher


def read_cv(file_path: str):
    text = ''
    with fitz.open(file_path) as doc:
        for page in doc:
            page_text = page.get_text()
            text = text + ' ' + page_text
    return text


def identify_sections(doc, nlp):
    matcher = Matcher(nlp.vocab)

    matcher.add("sections", [[{"ENT_TYPE":"HEADER"}]])
    matches = matcher(doc)
    sections = []
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        if not span.text.islower():
            sections.append((span.text, start))

    sections_pd = pd.DataFrame(sections, columns=['Header','Start'])
    sections_pd['NextSection'] = sections_pd['Start'].shift(-1).fillna(len(doc))
    return sections_pd


#choose information to retrieve
def get_section(sections_pd, section_name: str):
    section_name=section_name.lower()
    i_start = sections_pd.loc[sections_pd.Header.str.lower()==section_name].Start.values[0] + 1
    i_end = sections_pd.loc[sections_pd.Header.str.lower()==section_name].NextSection.values[0]
    txt = doc[i_start:i_end].text
    res = []
    for x in re.split('\n', txt):
        if len(x):
            if x.lower() == x:
                res[-1] = res[-1] + x
            else:
                res.append(x)
    return res


nlp = spacy.load("en_core_web_lg")
headers = "cv_headers.jsonl"
ruler = nlp.add_pipe('entity_ruler',before='ner')
ruler.from_disk(headers)

text=read_cv('cv5.pdf')
doc = nlp(text)
sections_pd = identify_sections(doc, nlp)

get_section(sections_pd, 'Skills')
get_section(sections_pd, 'Languages')
get_section(sections_pd, 'Hobbies')

# From fakespot-ai/roberta-base-ai-text-detection-v1 (Apache-2.0)
import re
from html import unescape


def clean_text(t):
    t = clean_markdown(t)
    t = t.replace("\n"," ")
    t = t.replace("\t"," ")
    t = t.replace("^M"," ")
    t = t.replace("\r"," ")
    t = t.replace(" ,", ",")
    t = re.sub(" +", " ", t)
    return t

def clean_markdown(md_text):
    # Remove code blocks
    md_text = re.sub(r'```.*?```', '', md_text, flags=re.DOTALL)
    # Remove inline code
    md_text = re.sub(r'`[^`]*`', '', md_text)
    # Remove images
    md_text = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
    # Remove links but keep link text
    md_text = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', md_text)
    # Remove bold and italic (groups of *, _)
    md_text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md_text)
    md_text = re.sub(r'(\*|_)(.*?)\1', r'\2', md_text)
    # Remove headings
    md_text = re.sub(r'#+ ', '', md_text)
    # Remove blockquotes
    md_text = re.sub(r'^>.*$', '', md_text, flags=re.MULTILINE)
    # Remove list markers
    md_text = re.sub(r'^(\s*[-*+]|\d+\.)\s+', '', md_text, flags=re.MULTILINE)
    # Remove horizontal rules
    md_text = re.sub(r'^\s*[-*_]{3,}\s*$', '', md_text, flags=re.MULTILINE)
    # Remove tables
    md_text = re.sub(r'\|.*?\|', '', md_text)
    # Remove raw HTML tags
    md_text = re.sub(r'<.*?>', '', md_text)
    # Decode HTML entities
    md_text = unescape(md_text)
    return md_text
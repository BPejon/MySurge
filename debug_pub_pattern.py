"""
Descobrir qual padrão está marcando [1] como publicação
"""

import re

seg = "Effects of foam microbubbles on electrical resistivity and capillary pressure of partially saturated porous media"

pub_patterns = [
    (r'\(\d{4}\)', 'ano_entre_parenteses'),
    (r'\b\d{4}\b', 'ano_isolado'),
    (r'vol\.?\s*\d+', 'volume'),
    (r'no\.?\s*\d+', 'numero'),
    (r'pp\.?\s*\d+', 'paginas_pp'),
    (r'pages?\s*\d+', 'paginas_pages'),
    (r'\d+\(\d+\):\d+--\d+', 'formato_comum'),
    (r'doi:\s*10\.\d{4,}', 'doi'),
    (r'https?://', 'url'),
    (r'technical report', 'tech_report'),
    (r'conference', 'conference'),
    (r'proceedings', 'proceedings'),
    (r'journal', 'journal'),
    (r'transactions', 'transactions'),
    (r'magazine', 'magazine'),
    (r'press', 'press'),
    (r'university', 'university'),
    (r'laboratory', 'laboratory'),
    (r'in:\s', 'in_keyword'),
]

seg_lower = seg.lower()

print(f"Testing segment:\n'{seg}'\n")

for pattern, name in pub_patterns:
    if re.search(pattern, seg_lower):
        print(f"✓ MATCH: {name} - pattern: {pattern}")
        match = re.search(pattern, seg_lower)
        print(f"  Found: '{match.group()}'")

print("\n\nNow testing the publication segment function:")
import sys
sys.path.insert(0, './src')
from markdownParser import is_publication_segment

result = is_publication_segment(seg)
print(f"is_publication_segment('{seg[:50]}...') = {result}")

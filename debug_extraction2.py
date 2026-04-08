"""
Debug detalhado da extração de títulos
"""

import sys
sys.path.insert(0, './src')

import markdownParser
import re

# Teste com referência [4] que está falhando
ref = "[4] Adebayo AR, Isah A, Mahmoud M, Al-Shehri D (2020). Effects of foam microbubbles on electrical resistivity and capillary pressure of partially saturated porous media. Transp Porous Media 133:405–424. https://doi.org/10.1007/s11242-020-01482-2"

print(f"Original ref: {ref}\n")

# Simular extract_title_from_ref
content = re.sub(r'^\[\d+\]\s*', '', ref).strip()
print(f"After ID removal: {content}\n")

# Dividir por pontos
segments = content.split('.')
segments = [seg.strip() for seg in segments if seg.strip()]

print(f"Segments ({len(segments)} total):")
for i, seg in enumerate(segments):
    is_pub = markdownParser.is_publication_segment(seg)
    is_author = markdownParser.is_author_segment(seg)
    print(f"  [{i}] {'[PUB]' if is_pub else ''} {'[AUTHOR]' if is_author else ''} {seg[:60]}...")

# Encontrar pub_start
pub_start = None
for i, seg in enumerate(segments):
    if markdownParser.is_publication_segment(seg):
        pub_start = i
        break

print(f"\npub_start = {pub_start}")

if pub_start is not None and pub_start > 1:
    title_segments = segments[1:pub_start]
    print(f"Title segments (1:{pub_start}): {title_segments}")
    final_title = '. '.join(title_segments)
    print(f"Final title: {final_title}")
elif pub_start is not None and pub_start == 1:
    print(f"pub_start == 1, checking if segments[0] is author: {markdownParser.is_author_segment(segments[0])}")
    if not markdownParser.is_author_segment(segments[0]):
        print(f"Not author, using segments[0]: {segments[0]}")

# Se ainda não achamos título
candidate = segments[1] if len(segments) > 1 else ""
print(f"\ncandidate from [1]: {candidate}")
print(f"is_author_segment: {markdownParser.is_author_segment(candidate)}")
print(f"is_publication_segment: {markdownParser.is_publication_segment(candidate)}")

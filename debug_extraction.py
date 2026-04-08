"""
Debug da extração de títulos
"""

import sys
sys.path.insert(0, './src')

import markdownParser

# Teste com a referência [3] que não foi encontrada
test_refs = [
    "[3] Adebayo AR, Bageri BS, Al Jaberi J, Salin RB (2020a). A calibration method for estimating mudcake thickness and porosity using NMR data. J Pet Sci Eng 195:107582",
    "[4] Adebayo AR, Isah A, Mahmoud M, Al-Shehri D (2020). Effects of foam microbubbles on electrical resistivity and capillary pressure of partially saturated porous media. Transp Porous Media 133:405–424. https://doi.org/10.1007/s11242-020-01482-2",
    "[22] org/10",  # Esta parece estar quebrada
    "[27] Baldwin B, Spinler E (1998). A direct method for simultaneously determining positive and negative capillary pressure curves in reservoir rock. SPE Form Eval 5:4–200",
]

print("Testing extract_title_from_ref:")
for ref in test_refs:
    title = markdownParser.extract_title_from_ref(ref)
    print(f"\nRef: {ref[:70]}...")
    print(f"Extracted: '{title}'")
    print(f"Length: {len(title)}")

# Agora vamos testar a normalização
from evaluator import normalize_string

print("\n\n" + "="*80)
print("Testing normalize_string:")

test_titles = [
    "A calibration method for estimating mudcake thickness and porosity using NMR data",
    "Effects of foam microbubbles on electrical resistivity and capillary pressure",
    "A direct method for simultaneously determining positive and negative capillary pressure curves in reservoir rock",
]

for title in test_titles:
    normalized = normalize_string(title)
    print(f"\nTitle: {title[:60]}...")
    print(f"Normalized: '{normalized}'")

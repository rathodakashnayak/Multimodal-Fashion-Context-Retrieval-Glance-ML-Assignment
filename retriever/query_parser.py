"""
query_parser.py
---------------
Decomposes a natural language fashion query into structured sub-attributes.

Why decompose?
--------------
Vanilla CLIP encodes the entire query as one vector. This means that for a
complex query like "red tie and white shirt in a formal setting", CLIP must
compress color, garment-type, AND location all into a single 512-d embedding.
This causes compositionality failures.

Our approach: split the query into *specialist sub-queries*, each targeting a
single attribute axis that CLIP handles well individually, then fuse the scores.

Attribute axes
--------------
  global     : the full original query (used as the baseline CLIP similarity)
  color      : dominant garment color(s)
  clothing   : garment type(s)
  location   : environment / setting
  style      : vibe / formality level

Design: rule-based extraction using keyword dictionaries. This is:
  - Offline (no LLM API key needed)
  - Deterministic and reproducible
  - Fast (<1ms per query)
  - Good enough for fashion domain vocabulary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ─────────────────────────── vocab dicts ─────────────────────────

# Map keywords → canonical attribute labels
COLOR_VOCAB: Dict[str, str] = {
    "red": "red",
    "blue": "blue",
    "navy": "navy blue",
    "green": "green",
    "olive": "olive green",
    "yellow": "yellow",
    "orange": "orange",
    "pink": "pink",
    "purple": "purple",
    "violet": "violet",
    "white": "white",
    "black": "black",
    "grey": "grey",
    "gray": "gray",
    "brown": "brown",
    "beige": "beige",
    "cream": "cream",
    "maroon": "maroon",
    "teal": "teal",
    "cyan": "cyan",
    "gold": "gold",
    "silver": "silver",
    "bright": "bright",
    "dark": "dark",
    "light": "light",
    "pastel": "pastel",
    "neon": "neon",
    "multicolor": "multicolor",
    "striped": "striped",
    "floral": "floral",
    "plaid": "plaid",
}

CLOTHING_VOCAB: Dict[str, str] = {
    # tops
    "shirt": "shirt",
    "t-shirt": "t-shirt",
    "tshirt": "t-shirt",
    "tee": "t-shirt",
    "blouse": "blouse",
    "top": "top",
    "sweater": "sweater",
    "hoodie": "hoodie",
    "sweatshirt": "sweatshirt",
    "tank top": "tank top",
    # bottoms
    "pants": "pants",
    "trousers": "trousers",
    "jeans": "jeans",
    "shorts": "shorts",
    "skirt": "skirt",
    "leggings": "leggings",
    # outerwear
    "jacket": "jacket",
    "coat": "coat",
    "blazer": "blazer",
    "raincoat": "raincoat",
    "parka": "parka",
    "vest": "vest",
    # dresses & full outfits
    "dress": "dress",
    "suit": "suit",
    "tuxedo": "tuxedo",
    "jumpsuit": "jumpsuit",
    "overalls": "overalls",
    "outfit": "outfit",
    # accessories
    "tie": "tie",
    "scarf": "scarf",
    "hat": "hat",
    "cap": "cap",
    "shoes": "shoes",
    "sneakers": "sneakers",
    "boots": "boots",
    "bag": "bag",
    "backpack": "backpack",
    # formality
    "formal": "formal attire",
    "casual": "casual wear",
    "business": "business attire",
    "sportswear": "sportswear",
    "athletic": "athletic wear",
    "workwear": "workwear",
}

LOCATION_VOCAB: Dict[str, str] = {
    "office": "office interior",
    "workplace": "office interior",
    "work": "office interior",
    "indoor": "indoor setting",
    "inside": "indoor setting",
    "street": "urban street",
    "city": "city street",
    "urban": "urban setting",
    "sidewalk": "urban street",
    "park": "park",
    "garden": "garden",
    "outdoor": "outdoor setting",
    "outside": "outdoor setting",
    "home": "home interior",
    "house": "home interior",
    "living room": "living room",
    "beach": "beach",
    "mall": "shopping mall",
    "restaurant": "restaurant",
    "gym": "gym",
    "studio": "studio",
    "airport": "airport",
    "bench": "park bench",
    "formal setting": "formal setting",
    "modern": "modern interior",
}

STYLE_VOCAB: Dict[str, str] = {
    "casual": "casual style",
    "formal": "formal style",
    "business": "business professional",
    "professional": "professional attire",
    "sporty": "sporty style",
    "athletic": "athletic style",
    "elegant": "elegant style",
    "chic": "chic style",
    "trendy": "trendy fashion",
    "classic": "classic style",
    "vintage": "vintage style",
    "streetwear": "streetwear",
    "bohemian": "bohemian style",
    "preppy": "preppy style",
    "weekend": "casual weekend",
    "party": "party attire",
    "summer": "summer fashion",
    "winter": "winter fashion",
}


# ─────────────────────────── data class ──────────────────────────

@dataclass
class ParsedQuery:
    """
    Structured representation of a decomposed fashion query.

    Attributes
    ----------
    global_query    : the original full query (passed to CLIP as-is)
    color           : extracted color descriptor or None
    clothing        : extracted garment type(s) or None
    location        : extracted environment/setting or None
    style           : extracted style/vibe or None

    sub_queries     : Dict[axis, str]  — single prompt per axis (legacy)
    ensemble_queries: Dict[axis, List[str]] — multiple CLIP-friendly
                      paraphrases per axis, used for prompt ensembling.
                      Averaging these embeddings gives much better coverage
                      than any single phrasing.
    """

    global_query: str
    color: Optional[str] = None
    clothing: Optional[str] = None
    location: Optional[str] = None
    style: Optional[str] = None
    sub_queries: Dict[str, str] = field(default_factory=dict)
    ensemble_queries: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ensemble_queries = self._build_ensemble_queries()
        # legacy single-string sub_queries for backward compat
        self.sub_queries = {k: v[0] for k, v in self.ensemble_queries.items()}

    def _build_ensemble_queries(self) -> Dict[str, List[str]]:
        """
        Build diverse CLIP-friendly text templates for each attribute axis.

        Each axis gets 4–6 paraphrases that emphasize different aspects
        of the same concept. The scorer averages their embeddings via
        encode_text_ensemble(), producing a more robust representation.

        Template design principles (from CLIP paper + fashion domain):
        - Include "a photo of" / "a photograph of" prefixes
        - Mix detailed and short descriptions
        - Include scene/activity context for location axis
        - Combine attributes in the composite axis
        """
        c  = self.color
        g  = self.clothing
        lo = self.location
        st = self.style
        q  = self.global_query

        queries: Dict[str, List[str]] = {}

        # ── global ──────────────────────────────────────────────────
        queries["global"] = [
            q,
            f"a photo of {q.lower().rstrip('.')}",
            f"fashion image: {q.lower().rstrip('.')}",
        ]

        # ── color + clothing (most discriminative axis) ───────────────
        if c and g:
            queries["color_clothing"] = [
                f"a person wearing a {c} {g}",
                f"a photo of someone in a {c} colored {g}",
                f"a {c} {g} outfit",
                f"a photograph of a person with a {c} {g}",
                f"close-up of a {c} {g} being worn",
                f"fashion photo: {c} {g}",
            ]
        elif c:
            queries["color"] = [
                f"clothing in {c} color",
                f"a person wearing {c} colored clothing",
                f"a photo of {c} fashion",
                f"{c} outfit",
                f"someone dressed in {c}",
            ]
        elif g:
            queries["clothing"] = [
                f"a person wearing a {g}",
                f"a photo of someone wearing a {g}",
                f"a {g} being worn",
                f"fashion photo of a {g}",
                f"a person dressed in a {g}",
            ]

        # ── location ──────────────────────────────────────────────────
        if lo:
            # Map canonical location to richer scene descriptions
            LOCATION_SCENES = {
                "park": [
                    "person sitting on a park bench outdoors",
                    "fashion photo in a green park",
                    "outdoor park setting with trees",
                    "someone relaxing in a public park",
                    "park bench outdoor scene",
                    "a photo taken in a park",
                ],
                "park bench": [
                    "person sitting on a park bench",
                    "outdoor bench in a park",
                    "someone seated on a bench outside",
                    "park bench scene",
                    "a photo of a person on a bench in a park",
                ],
                "office interior": [
                    "professional setting inside a modern office",
                    "person in an office environment",
                    "office building interior with desks",
                    "corporate office fashion photo",
                    "a photo taken inside an office",
                ],
                "city street": [
                    "person walking on a city street",
                    "urban street fashion photo",
                    "outdoor city sidewalk scene",
                    "a photo on a busy city street",
                    "street style fashion in the city",
                ],
                "urban street": [
                    "urban street scene with fashion",
                    "person on a city sidewalk",
                    "streetwear photo outdoors in city",
                ],
                "formal setting": [
                    "formal event venue interior",
                    "a person in a formal setting",
                    "elegant formal occasion",
                    "professional formal environment",
                ],
                "modern interior": [
                    "stylish modern interior space",
                    "contemporary office or building interior",
                    "modern room with clean design",
                ],
            }
            queries["location"] = LOCATION_SCENES.get(lo, [
                f"fashion photo in {lo}",
                f"a person photographed in {lo}",
                f"a photo taken at {lo}",
                f"outdoor/indoor scene: {lo}",
            ])

        # ── style ─────────────────────────────────────────────────────
        if st:
            queries["style"] = [
                f"person dressed in {st}",
                f"a photo showing {st} fashion",
                f"{st} outfit style",
                f"a {st} look",
            ]

        # ── composite: all attributes together ────────────────────────
        # When multiple attributes are detected, build a combined prompt
        # that captures the full scene description.
        parts = []
        if c and g:
            parts.append(f"{c} {g}")
        elif c:
            parts.append(f"{c} clothing")
        elif g:
            parts.append(f"{g}")
        if lo:
            parts.append(f"in {lo}")
        if st:
            parts.append(f"({st} style)")

        if len(parts) >= 2:  # only add composite when we have multiple attrs
            composite_desc = " ".join(parts)
            queries["composite"] = [
                f"a person wearing {composite_desc}",
                f"a photo of someone with {composite_desc}",
                f"fashion image: {composite_desc}",
                f"a person photographed {composite_desc}",
            ]

        return queries

    def summary(self) -> str:
        parts = [f"Global: '{self.global_query}'"]
        if self.color:
            parts.append(f"Color: {self.color}")
        if self.clothing:
            parts.append(f"Clothing: {self.clothing}")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.style:
            parts.append(f"Style: {self.style}")
        return " | ".join(parts)


# ─────────────────────────── parser ──────────────────────────────

class QueryParser:
    """
    Rule-based decomposition of natural language queries.

    The parser scans the query string for known vocabulary terms and
    assembles a ParsedQuery with structured ensemble sub-queries for each axis.
    """

    def parse(self, query: str) -> ParsedQuery:
        q_lower = query.lower()

        color = self._extract(q_lower, COLOR_VOCAB)
        clothing = self._extract_clothing(q_lower)
        location = self._extract(q_lower, LOCATION_VOCAB)
        style = self._extract(q_lower, STYLE_VOCAB)

        return ParsedQuery(
            global_query=query,
            color=color,
            clothing=clothing,
            location=location,
            style=style,
        )

    # ── internal ───────────────────────────────────────────────────

    @staticmethod
    def _extract(text: str, vocab: Dict[str, str]) -> Optional[str]:
        """Return the first matching canonical value from vocab, or None."""
        for keyword, canonical in vocab.items():
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text):
                return canonical
        return None

    @staticmethod
    def _extract_clothing(text: str) -> Optional[str]:
        """
        Extract garment type. Multi-word terms checked before single-word ones.
        """
        sorted_vocab = sorted(CLOTHING_VOCAB.items(), key=lambda x: -len(x[0]))
        for keyword, canonical in sorted_vocab:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text):
                return canonical
        return None


# ─────────────────────────── quick test ──────────────────────────

if __name__ == "__main__":
    parser = QueryParser()
    test_queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ]
    print("\nQuery Parser Test\n" + "="*60)
    for q in test_queries:
        parsed = parser.parse(q)
        print(f"\n> {parsed.summary()}")
        for axis, templates in parsed.ensemble_queries.items():
            print(f"  [{axis}] ({len(templates)} templates):")
            for t in templates[:3]:
                print(f"    - {t}")


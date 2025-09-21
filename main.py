from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import unidecode

class Mapper:
    def __init__(self, alias_map: Dict[str, List[str]], icd_entries: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.alias_map = alias_map
        self.icd_entries = icd_entries
        # Precompute ICD embeddings
        self.icd_embeddings = []
        for icd in icd_entries:
            icd_texts = self._build_icd_variants(icd)
            emb = self.model.encode(icd_texts, convert_to_tensor=True)
            self.icd_embeddings.append(emb)

    def normalize_text(self, text: str) -> str:
        """
        Normalize text: remove diacritics, lowercase, strip punctuation.
        """
        # 1. Unicode normalization (remove accents/diacritics)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        text = text.lower()
        text = re.sub(r"[\(\)\[\]\{\},.;:!?]", " ", text) # Remove brackets, punctuation except needed hyphens
        text = re.sub(r"\s+", " ", text).strip() # extra spaces
        
        return text
    
    def tokenize(self, text, phrase_join=True):
        """
        Custom tokenizer for NAMASTE ↔ ICD-11 TM2 terms.
        - Normalizes input
        - Removes stopwords
        - Optionally joins medical phrases with underscores
        """
        STOPWORDS = {"due", "to", "of", "and", "or", "the", "a", "an", "with", "in"}
        text = normalize_text(text)
        
        words = text.split()
        words = [w for w in words if w not in STOPWORDS]
        
        # Phrase joining: "migraine disorder" -> "migraine_disorder"
        if phrase_join:
            tokens = []
            skip_next = False
            for i, w in enumerate(words):
                if len(w)<=2 or w=='tm2': continue
                if skip_next:
                    skip_next = False
                    continue
                if i < len(words)-1 and words[i+1] in ["disorder", "disease", "syndrome"]:
                    tokens.append(f"{w}_{words[i+1]}")
                    skip_next = True
                else:
                    tokens.append(w)
            return tokens
        return words
        
    def _build_namaste_variants(self, entry: Dict) -> List[str]:
        variants = []
        # title
        title_norm = self.normalize_text(entry.get('title', ''))
        variants.append(title_norm)
        # synonyms
        for syn in entry.get('synonyms', []):
            variants.append(self.normalize_text(syn))
        # aliases
        if title_norm in self.alias_map:
            for alias in self.alias_map[title_norm]:
                variants.append(self.normalize_text(alias))
        return variants

    def _build_icd_variants(self, icd: Dict) -> List[str]:
        variants = []
        variants.append(self.normalize_text(icd.get('title', '')))
        for syn in icd.get('synonyms', []):
            variants.append(self.normalize_text(syn))
        return variants

    def map_entry(self, namaste_entry: Dict, top_k: int = 3) -> List[Dict]:
        # Encode Namaste variants
        namaste_texts = self._build_namaste_variants(namaste_entry)
        namaste_emb = self.model.encode(namaste_texts, convert_to_tensor=True)
        namaste_emb = namaste_emb / namaste_emb.norm(p=2, dim=-1, keepdim=True)

        results = []
        for icd, icd_emb in zip(self.icd_entries, self.icd_embeddings):
            # Compute cosine similarity
            sim_matrix = util.cos_sim(namaste_emb, icd_emb).cpu().numpy()
            score = float(np.max(sim_matrix))
            results.append({
                "icd_code": icd['code'],
                "title":icd['title'],
                "similarity": score
            })

        # Sort top_k
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
        return results


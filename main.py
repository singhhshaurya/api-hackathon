from typing import List, Dict, Any, Tuple
import re
import json
import os
import numpy as np

# Install if missing:
# pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer, util

# Optional FAISS import (fall back gracefully if not installed) REMOVED.

# -- Normalization helpers --

def normalize_text(text: str) -> str:
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

def tokenize(text, phrase_join=True):
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


def expand_aliases_for_entry(namaste_entry, alias_dict):
    """
    For a NAMASTE entry, build alias expansions from tokens (uses sanskrit to english aliases.)
    Returns list of alias phrases added.
    """
    aliases = set()
    fields = []
    if 'title' in namaste_entry:
        fields.append(namaste_entry['title'])
    if 'synonyms' in namaste_entry:
        fields.extend(namaste_entry['synonyms'])
    if 'description' in namaste_entry:
        fields.append(namaste_entry['description'])
    text = " ".join([str(x) for x in fields if x])
    tokens = tokenize(text)
    for t in tokens:
        if t in alias_dict:
            for a in alias_dict[t]:
                aliases.add(a)
    print(list(aliases))
    return list(aliases)

# -- Embedding helpers --
# embedding is just a way of turning text (words, phrases, sentences) into a list of numbers (vector).

class Embedder:
    def __init__(self, model_name= "paraphrase-multilingual-mpnet-base-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        if len(sentences) == 0:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        embs = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return embs

        
# -- Mapper --
class Mapper:
    def __init__(self,
                 icd_entries: List[Dict[str, Any]],
                 alias_dict: Dict[str, List[str]],
                 model_name: str = "paraphrase-multilingual-mpnet-base-v2",
                 weights: Dict[str, float] = None,
                 use_faiss: bool = False,
                 faiss_index_path: str = None,
                 device: str = "cpu"):
        """
        icd_entries: list of ICD dicts with keys: 'code', 'title', 'synonyms' (list)
        alias_dict: mapping sanaskrit_token -> list of english alias strings
        weights: aggregation field weights, e.g. {'title':0.5,'syn':0.3,'desc':0.15,'alias':0.05}
        """
        self.icd_entries = icd_entries
        self.alias_dict = alias_dict
        self.embedder = Embedder(model_name=model_name, device=device)
        self.weights = weights or {'title':0.5, 'syn':0.3, 'desc':0.15, 'alias':0.05}
        self._prepare_icd_corpus()
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index = None
        self.faiss_index_path = faiss_index_path
        if self.use_faiss:
            self._build_faiss_index()

    def _prepare_icd_corpus(self):
        """
        Prepare structured text for ICD entries and compute embeddings by field.
        """
        self.icd_meta = []
        title_texts = []
        syn_texts = []
        desc_texts = []  # often missing for ICD small entries
        alias_texts = []

        for ent in self.icd_entries:
            code = ent.get('code')
            title = ent.get('title','')
            syns = ent.get('synonyms', [])
            desc = ent.get('definition', '') or ent.get('description','')

            norm_title = normalize_text(title)
            norm_syn = "; ".join([normalize_text(s) for s in syns]) if syns else ""
            norm_desc = normalize_text(desc)

            # create aliases for ICD as well (optional)
            icd_aliases = []  # may remain empty unless you provide mapping for ICD terms

            self.icd_meta.append({
                'code': code,
                'title': title,
                'synonyms': syns,
                'norm_title': norm_title,
                'norm_syn': norm_syn,
                'norm_desc': norm_desc,
                'aliases': icd_aliases
            })
            title_texts.append(f"[CODE] {code} [TITLE] {norm_title}")
            syn_texts.append(f"[SYN] {norm_syn}")
            desc_texts.append(f"[DESC] {norm_desc}")
            alias_texts.append("; ".join(icd_aliases) if icd_aliases else "")

        # embed fields
        self.icd_title_embs = self.embedder.embed_sentences(title_texts)
        self.icd_syn_embs = self.embedder.embed_sentences(syn_texts)
        self.icd_desc_embs = self.embedder.embed_sentences(desc_texts)
        self.icd_alias_embs = self.embedder.embed_sentences(alias_texts)

    def _aggregate_icd_vector(self, idx: int) -> np.ndarray:
        """
        Aggregate field-level embeddings into a single vector using weights.
        """
        w = self.weights
        vec = np.zeros_like(self.icd_title_embs[0])
        vec += w.get('title',0.0) * self.icd_title_embs[idx]
        vec += w.get('syn',0.0) * self.icd_syn_embs[idx]
        vec += w.get('desc',0.0) * self.icd_desc_embs[idx]
        vec += w.get('alias',0.0) * self.icd_alias_embs[idx]
        # normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


    def save(self, path: str):
        """
        Save metadata and optionally FAISS index & embeddings
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "icd_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.icd_meta, f, ensure_ascii=False, indent=2)
        if self.use_faiss and self.faiss_index is not None and self.faiss_index_path:
            faiss.write_index(self.faiss_index, self.faiss_index_path)

    # ---------- search utilities ----------

    def _embed_namaste(self, nam: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Build field-level normalized texts and embeddings for a NAMASTE entry.
        Returns dict with fields: title_emb, syn_emb, desc_emb, alias_emb
        """
        print("DORA")
        title = normalize_text(nam.get('title',''))
        syn = "; ".join([normalize_text(s) for s in nam.get('synonyms',[])]) if nam.get('synonyms') else ""
        desc = normalize_text(nam.get('description',''))
        aliases = expand_aliases_for_entry(nam, self.alias_dict)
        alias_text = "; ".join(aliases) if aliases else ""

        title_text = f"[TITLE] {title}"
        syn_text = f"[SYN] {syn}"
        desc_text = f"[DESC] {desc}"
        alias_text_full = f"[ALIASES] {alias_text}"

        emb_title = self.embedder.embed_sentences([title_text])[0]
        emb_syn = self.embedder.embed_sentences([syn_text])[0] if syn_text else np.zeros_like(emb_title)
        emb_desc = self.embedder.embed_sentences([desc_text])[0] if desc_text else np.zeros_like(emb_title)
        emb_alias = self.embedder.embed_sentences([alias_text_full])[0] if alias_text else np.zeros_like(emb_title)

        return {
            'norm_title': title,
            'norm_syn': syn,
            'norm_desc': desc,
            'aliases': aliases,
            'title_emb': emb_title,
            'syn_emb': emb_syn,
            'desc_emb': emb_desc,
            'alias_emb': emb_alias
        }

    def _score_against_icd(self, nam_embs: Dict[str, np.ndarray], top_k: int = 10) -> List[Dict[str,Any]]:
        """
        Compute similarity against all ICD entries (fast path via FAISS if available).
        Returns top_k candidate dicts with breakdowns and aggregated score.
        """
        n_icd = len(self.icd_meta)
        
        #  compute full pairwise (ok for small-medium ICD sizes)
        title_sims = util.cos_sim(nam_embs['title_emb'], self.icd_title_embs).cpu().numpy().reshape(-1)
        syn_sims = util.cos_sim(nam_embs['syn_emb'], self.icd_syn_embs).cpu().numpy().reshape(-1)
        desc_sims = util.cos_sim(nam_embs['desc_emb'], self.icd_desc_embs).cpu().numpy().reshape(-1)
        alias_sims = util.cos_sim(nam_embs['alias_emb'], self.icd_alias_embs).cpu().numpy().reshape(-1)

        agg_scores = (self.weights.get('title',0.0)*title_sims +
                      self.weights.get('syn',0.0)*syn_sims +
                      self.weights.get('desc',0.0)*desc_sims +
                      self.weights.get('alias',0.0)*alias_sims)
        idxs = np.argsort(-agg_scores)[:top_k]
        candidates = []
        for i in idxs:
            candidates.append({
                'icd_idx': int(i),
                'code': self.icd_meta[i]['code'],
                'title': self.icd_meta[i]['title'],
                'sim_agg': float(agg_scores[i])
                # 'sim_breakdown': {'title': float(title_sims[i]), 'syn': float(syn_sims[i]), 'desc': float(desc_sims[i]), 'alias': float(alias_sims[i])}
            })
        return candidates

    def explain_match(self, nam: Dict[str,Any], candidate: Dict[str,Any]) -> Dict[str,Any]:
        """
        Build explainability dict: token overlaps, alias hits, top-matching synonym text.
        """
        nam_tokens = set(tokenize(nam.get('title','') + " " + " ".join(nam.get('synonyms',[])) + " " + nam.get('description','') if nam.get('description') else ""))
        icd = self.icd_meta[candidate['icd_idx']]
        icd_text = icd.get('norm_title','') + " " + icd.get('norm_syn','') + " " + icd.get('norm_desc','')
        icd_tokens = set(tokenize(icd_text))
        overlap = nam_tokens.intersection(icd_tokens)
        alias_hits = [a for a in expand_aliases_for_entry(nam, self.alias_dict) if any(t in icd_text for t in tokenize(a))]
        return {
            'token_overlap': list(overlap),
            'num_overlap': len(overlap),
            'alias_hits': alias_hits,
            'sim_breakdown': candidate.get('sim_breakdown',{})
        }

    def map_single(self, nam: Dict[str,Any], top_k: int = 3, accept_thr: float = 0.72, review_thr: float = 0.55) -> Dict[str,Any]:
        """
        Map a single NAMASTE entry to ICD candidates with accept/review/no-match label and explanations.
        """
        nam_embs = self._embed_namaste(nam)
        candidates = self._score_against_icd(nam_embs, top_k=top_k)
        if len(candidates) == 0:
            return {'top_k': [], 'label': 'no-match'}

        best = candidates[0]
        label = 'no-match'
        if best['sim_agg'] >= accept_thr:
            label = 'accepted'
        elif best['sim_agg'] >= review_thr:
            label = 'review'
        else:
            label = 'no-match'

        explain = self.explain_match(nam, best)
        return {
            'namaste_title': nam.get('title'),
            'top_k': candidates,
            'predicted': {'code': best['code'], 'title': best['title'], 'score': best['sim_agg']},
            'label': label,
            'explain': explain
        }

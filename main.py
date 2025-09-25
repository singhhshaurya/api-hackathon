# ============================================================================
# NAMASTE to ICD-11 TM2 Mapping System
# ============================================================================
# This module provides functionality to map NAMASTE Ayurveda diagnostic codes
# to corresponding ICD-11 TM2 Traditional Medicine codes using semantic similarity.

# ============================================================================
# IMPORTS
# ============================================================================

# Third-party imports
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Standard library imports
import re
import unicodedata
import json
import uuid
from datetime import date, datetime, timezone

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Similarity threshold constants
EXACT_TH = 0.7          # Threshold for exact matches
RELATED_TH = 0.3        # Threshold for related matches

# System URIs for FHIR resources
NAMASTE_SYSTEM = "http://namaste.in/codes"
ICD11_SYSTEM = "http://id.who.int/icd/release/11/tm2"

# ============================================================================
# IN-MEMORY STATE MANAGEMENT
# ============================================================================
# NOTE: These are not persistent across restarts
# TODO: Future Plan - Migrate to a database or create CodeSystem and ConceptMap 
#       resources when we have a supervised model with human-in-the-loop feedback

# Index for quick lookup of concept mappings
concept_map_index: dict = {}

# FHIR ConceptMap resource structure
concept_map = {
    "resourceType": "ConceptMap",
    "id": "namaste-to-tm2",
    "name": "NAMASTEtoTM2",
    "description": "This ConceptMap maps standardized NAMASTE Ayurveda diagnostic codes to corresponding ICD-11 TM2 Traditional Medicine codes for clinical interoperability.",
    "status": "active",
    "date":date.today().isoformat(),
    "sourceUri": "http://namaste.org/codes",
    "targetUri": "http://id.who.int/icd/release/11/2025-01/tm2",
    "group": [
        {
            'element':[]
        }
    ]
}


# ============================================================================
# DATA LOADING
# ============================================================================

# Load ICD-11 TM2 data
with open("icd11_data.json", "r", encoding="utf-8") as f:
    icd_list = json.load(f)

# Load Sanskrit to English translation mappings
with open("sanskrit-to-english.json", "r", encoding="utf-8") as f:
    alias = json.load(f)

# Load NAMASTE diagnostic entries
with open("namaste_entries.json", "r", encoding="utf-8") as f:
    namaste_data_mapped = json.load(f)
    
#FHIR CodeSystem resource structure
codesystem_namaste = {
    "resourceType": "CodeSystem",
    "id": "NAMASTE",
    "text": {
        "status": "generated",
        "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\"><p class=\"res-header-id\"><b>Generated Narrative: CodeSystem NAMASTE</b></p><a name=\"NAMASTE\"> </a><p>This case-sensitive code system <code>http://namaste.in/codesystem</code> defines codes, but no codes are represented here</p></div>"
    },
    "url": "http://namaste.in/codesystem",
    "version": "1.0.0",
    "name": "NAMASTE",
    "title": "NAMASTE Code System for Traditional Medicine in India",
    "status": "active",
    "experimental": "false",
    "date": "2025-09-22T00:00:00-00:00",
    "publisher": "Ministry of AYUSH, India",
    "contact": [
        {
        "name": "Ministry of AYUSH, Government of India",
        "telecom": [
            {
            "system": "url",
            "value": "https://namaste.ayush.gov.in"
            }
        ]
        }
    ],
    "description": "The NAMASTE code system defines codes for traditional medicine conditions in India. It provides a standard structure for Ayurvedic and other traditional medicine terms, enabling FHIR-compliant integration and mapping to ICD-11 TM2 codes. More information can be found at [https://namaste.ayush.gov.in](https://namaste.ayush.gov.in).",
    "copyright": "Ministry of AYUSH, Government of India. All rights reserved.",
    "caseSensitive": True,
    "count": len(namaste_data_mapped),
    "concept":[namaste_data_mapped]
}

codesystem_icd = {
    "resourceType" : "CodeSystem",
    "id" : "ICD11MMS",
    "text" : {
        "status" : "generated",
        "div" : "<div xmlns=\"http://www.w3.org/1999/xhtml\"><p class=\"res-header-id\"><b>Generated Narrative: CodeSystem ICD11MMS</b></p><a name=\"ICD11MMS\"> </a><a name=\"hcICD11MMS\"> </a><p>This case-sensitive code system <code>http://id.who.int/icd/release/11/mms</code> defines codes, but no codes are represented here</p></div>"
    },
    "url" : "http://id.who.int/icd/release/11/mms",
    "version" : "1.0.0",
    "name" : "ICD11MMS",
    "title" : "International Classification of Diseases, 11th Revision Mortality and Morbidity Statistics (MMS)",
    "status" : "active",
    "experimental" : "false",
    "date" : "2022-11-15T00:00:00-00:00",
    "publisher" : "The World Health Organization",
    "contact" : [
        {
        "name" : "The World Health Organization; 20, avenue Appia 1211 Geneva 27, Switzerland",
        "telecom" : [
            {
            "system" : "url",
            "value" : "https://icd.who.int/en"
            }
        ]
        }
    ],
    "description" : "The International Classification of Diseases, 11th Revision Mortality and Morbidity Statistics (MMS) is one of the ICD11 linearizations. Information about the ICD Foundation Component and the ICD11 Linearizations can be found in the complete reference guide here: [https://icd.who.int/icd11refguide/en/index.html](https://icd.who.int/icd11refguide/en/index.html)\r\n\r\n\"**The ICD11 Linearizations (Tabular lists)**\r\n\r\nA linearization is a subset of the foundation component, that is:\r\n\r\n1. fit for a particular purpose: reporting mortality, morbidity, primary care or other uses;\r\n\r\n 2. composed of entities that are Mutually Exclusive of each other; \r\n\r\n3. each entity is given a single parent.\r\n\r\nLinearization is similar to the classical print versions of ICD Tabular List (e.g. volume I of ICD-10 or other previous editions). The main linearization of ICD-11 is Mortality and Morbidity Statistics (MMS). Various linearizations could be built at different granularity, use case or other purposes such as for Primary Care, Clinical Care or Research. The linkage from the foundation component to a particular linearization will ensure consistent use of the ICD.\"\r\n\r\nICD-11 for Mortality and Morbidity (ICD-11 MMS) can be downloaded in either print or electronic (spreadsheet) format from the  browser in the Info tab located [here](https://icd.who.int/browse11/l-m/en)",
    "copyright" : "The WHO grants a license for \"commercial and non-commercial use\" of ICD-11\r\n\r\nCC BY-ND 3.0 IGO\r\n\r\nDetailed information can be found here: [https://icd.who.int/en/docs/icd11-license.pdf](https://icd.who.int/en/docs/icd11-license.pdf)\r\n\r\nContact licensing@who.int to obtain further information.",
    "caseSensitive" : "true",
    "content" : "complete",
    "count": len(icd_list),
    "concept": [icd_list]
}
# ============================================================================
# SEMANTIC MAPPING CLASS
# ============================================================================

class Mapper:
    """
    Semantic mapping system for NAMASTE to ICD-11 TM2 code translation.
    
    This class uses sentence transformers to compute semantic similarity between
    NAMASTE Ayurveda diagnostic terms and ICD-11 Traditional Medicine codes.
    """
    
    def __init__(self, alias_map, icd_entries, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the mapper with alias mappings and ICD entries.
        
        Args:
            alias_map (dict): Sanskrit to English translation mappings
            icd_entries (list): List of ICD-11 TM2 entries with codes and descriptions
            model_name (str): Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.alias_map = alias_map
        self.icd_entries = icd_entries
        
        # Precompute ICD embeddings for faster similarity computation
        self.icd_embeddings = []
        for icd in icd_entries:
            icd_texts = self._build_icd_variants(icd)
            emb = self.model.encode(icd_texts, convert_to_tensor=True)
            self.icd_embeddings.append(emb)

    def normalize_text(self, text):
        """
        Normalize text for consistent processing.
        
        Performs Unicode normalization, converts to lowercase, removes punctuation,
        and cleans up extra whitespace.
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        # Unicode normalization (remove accents/diacritics)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove brackets and punctuation (keep hyphens for medical terms)
        text = re.sub(r"[\(\)\[\]\{\},.;:!?]", " ", text)
        
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def tokenize(self, text, phrase_join=True):
        """
        Custom tokenizer for medical terminology.
        
        Normalizes input, removes stopwords, and optionally joins medical phrases
        with underscores for better semantic matching.
        
        Args:
            text (str): Input text to tokenize
            phrase_join (bool): Whether to join medical phrases (e.g., "migraine disorder")
            
        Returns:
            list: List of processed tokens
        """
        # Define medical stopwords to exclude
        STOPWORDS = {"due", "to", "of", "and", "or", "the", "a", "an", "with", "in"}
        
        text = self.normalize_text(text)
        words = text.split()
        words = [w for w in words if w not in STOPWORDS]
        
        if phrase_join:
            tokens = []
            skip_next = False
            
            for i, w in enumerate(words):
                # Skip very short words and 'tm2'
                if len(w) <= 2 or w == 'tm2':
                    continue
                    
                if skip_next:
                    skip_next = False
                    continue
                    
                # Join medical terminology phrases
                if i < len(words) - 1 and words[i + 1] in ["disorder", "disease", "syndrome"]:
                    tokens.append(f"{w}_{words[i + 1]}")
                    skip_next = True
                else:
                    tokens.append(w)
                    
            return tokens
        
        return words
        
    def _build_namaste_variants(self, entry):
        """
        Build text variations for a NAMASTE entry to improve embedding robustness.
        
        Combines the main title, synonyms, and Sanskrit-to-English aliases to create
        multiple text variants for better semantic matching.
        
        Args:
            entry (dict): NAMASTE entry with title, synonyms, etc.
            
        Returns:
            list: List of text variants for the entry
        """
        variants = []
        
        # Process main title
        title_norm = self.normalize_text(entry.get('title', ''))
        variants += self.tokenize(entry.get('title', ''))
        
        # Process synonyms
        for syn in entry.get('synonyms', []):
            variants += self.tokenize(syn)
            
        # Process Sanskrit-to-English aliases
        if title_norm in self.alias_map:
            for alias in self.alias_map[title_norm]:
                variants += self.tokenize(alias)
                
        return variants

    def _build_icd_variants(self, icd):
        """
        Build text variations for an ICD-11 entry.
        
        Args:
            icd (dict): ICD-11 entry with title and synonyms
            
        Returns:
            list: List of normalized text variants
        """
        variants = []
        variants.append(self.normalize_text(icd.get('title', '')))
        for syn in icd.get('synonyms', []):
            variants.append(self.normalize_text(syn))
        return variants

    def map_entry(self, namaste_entry, top_k=3):
        """
        Map a NAMASTE entry to the most similar ICD-11 TM2 codes.
        
        Uses semantic similarity computation between NAMASTE text variants
        and precomputed ICD-11 embeddings.
        
        Args:
            namaste_entry (dict): NAMASTE entry to map
            top_k (int): Number of top matches to return
            
        Returns:
            list: Top k(we will use k=3) ICD-11 matches with similarity scores
        """
        # Encode NAMASTE variants into embeddings
        namaste_texts = self._build_namaste_variants(namaste_entry)
        namaste_emb = self.model.encode(namaste_texts, convert_to_tensor=True)
        
        # Normalize embeddings for cosine similarity
        namaste_emb = namaste_emb / namaste_emb.norm(p=2, dim=-1, keepdim=True)

        results = []
        for icd, icd_emb in zip(self.icd_entries, self.icd_embeddings):
            # Compute cosine similarity matrix
            sim_matrix = util.cos_sim(namaste_emb, icd_emb).cpu().numpy()
            score = float(np.max(sim_matrix))
            
            results.append({
                "icd_code": icd['code'],
                "title": icd['title'],
                "similarity": score
            })

        # Return top k results sorted by similarity
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
        return results


# ============================================================================
# MAPPER INITIALIZATION
# ============================================================================

# Initialize the global mapper instance
mapper = Mapper(alias, icd_list)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _category_for_score(score: float):
    """
    Categorize mapping quality based on similarity score.
    
    Args:
        score (float): Similarity score between 0 and 1
        
    Returns:
        str: Category string for FHIR ConceptMap equivalence
    """
    if score >= EXACT_TH:
        return "exactMatch"
    if score >= RELATED_TH:
        return "relatedMatch"
    return "noMatch"


def add_to_concept_map(namaste_entry, top_matches):
    """
    Format mapping results for FHIR ConceptMap structure.
    
    Args:
        namaste_entry (dict): Source NAMASTE entry
        top_matches (list): List of ICD-11 matches with similarity scores
        
    Returns:
        dict: ConceptMap element structure
    """
    entry = {
        "code": namaste_entry['code'],
        "display": namaste_entry['title'],
        "target": []
    }
    
    for match in top_matches:
        target = {
            'code': match['icd_code'],
            'display': match['title'],
            "equivalence": _category_for_score(match['similarity']),
            "similarityScore": match['similarity']
        }
        entry['target'].append(target)
    
    return entry


def fetch_namaste_entry(namaste_code):
    """
    Retrieve a NAMASTE entry by its code.
    
    Args:
        namaste_code (str): NAMASTE diagnostic code
        
    Returns:
        dict or None: NAMASTE entry if found, None otherwise
    """
    return namaste_data_mapped.get(namaste_code)


# ============================================================================
# CORE MAPPING FUNCTIONS
# ============================================================================

def namaste_to_icd(namaste_code):
    """
    Map a NAMASTE code to ICD-11 TM2 codes with caching.
    
    Checks ConceptMap first, then performs semantic mapping if not found.
    Results are added to ConceptMap resource for future lookups.
    
    Args:
        namaste_code (str): NAMASTE diagnostic code
        
    Returns:
        dict or None: ConceptMap element with ICD-11 mappings, or None if not found
    """
    # Check conceptmap first
    if namaste_code in concept_map_index:
        index = concept_map_index[namaste_code]
        return concept_map['group'][0]['element'][index]
        
    # Fetch NAMASTE entry
    namaste_entry = fetch_namaste_entry(namaste_code)
    
    if namaste_entry:
        # Perform semantic mapping
        top_matches = mapper.map_entry(namaste_entry)
        mapped_entry = add_to_concept_map(namaste_entry, top_matches)
        
        # Cache the result
        concept_map['group'][0]['element'].append(mapped_entry)
        concept_map_index[namaste_code] = len(concept_map['group'][0]['element']) - 1
        
        return mapped_entry
    
    return None


# ============================================================================
# FHIR RESOURCE CREATION
# ============================================================================

def make_fhir_condition(
    namaste_code,       
    subject_id=None,   
    resource_id=None, 
    timestamp=None     
):
    """
    Create a FHIR Condition resource from a NAMASTE diagnostic code.
    
    Maps the NAMASTE code to ICD-11 TM2 codes and creates a properly formatted
    FHIR Condition resource with both coding systems.
    
    Args:
        namaste_code (str): NAMASTE diagnostic code
        subject_id (str, optional): Patient/subject reference
        resource_id (str, optional): Resource ID (UUID generated if not provided)
        timestamp (str, optional): ISO timestamp (current time if not provided)
        
    Returns:
        str: JSON string of the FHIR Condition resource
    """
    # Get mappings from NAMASTE to ICD-11
    mappings = namaste_to_icd(namaste_code)
    namaste_entry = namaste_data_mapped[namaste_code]
    namaste_display = namaste_entry['title']
    namaste_description = namaste_entry['description']

    # Generate defaults if not provided
    if resource_id is None:
        resource_id = str(uuid.uuid4())

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    

    # Build coding array with NAMASTE and ICD-11 codes
    coding = []
    coding.append({
        "system": NAMASTE_SYSTEM,
        "code": namaste_code,
        "display": namaste_display
    })
    
    # Collect mapping extensions for metadata
    extensions = []
    
    if mappings and 'target' in mappings:
        for idx, target in enumerate(mappings['target']):
            icd_code = target.get("code")
            icd_display = target.get("display")
            equivalence = target.get("equivalence")
            similarityscore = target.get("similarityScore")
            
            # Add to coding array
            coding.append({
                "system": ICD11_SYSTEM,
                "code": icd_code,
                "display": icd_display
            })
            
            # Add extension metadata
            extensions.append({
                "candidateIndex": idx,
                "system": ICD11_SYSTEM,
                "code": icd_code,
                "display": icd_display,
                "mappingCategory": equivalence,
                "similarityScore": round(similarityscore, 4)
            })

    # Determine verification status based on best match
    verification_status = "provisional"  # Default
    if mappings and 'target' in mappings and mappings['target']:
        best_equivalence = mappings['target'][0].get('equivalence', 'noMatch')
        if best_equivalence == "exactMatch":
            verification_status = "confirmed"

    # Build FHIR Condition resource
    condition = {
        "resourceType": "Condition",
        "id": resource_id,
        "meta": {
            "lastUpdated": timestamp
        },
        "code": {
            "coding": coding[:2],
            "text": f"{namaste_display} mapped to ICD-11 candidates ({coding[1]['code']}, {coding[2]['code']}, {coding[3]['code']})"
        },
        "verificationStatus": {"coding": [{"code": verification_status}]},
        "extension":[
            {
                "url": "http://namaste.in/fhir/StructureDefinition/mappingCandidates",
                "valueMappingCandidates": extensions
            }
        ]
    }
    
    # Add subject if provided
    if subject_id:
        condition["subject"] = {"reference": f"Patient/{subject_id}"}
    
    return json.dumps(condition, indent=2, ensure_ascii=False)





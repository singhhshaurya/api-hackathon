import json
import uuid
from datetime import datetime
# import namaste_data_code_mapping from initialiser
# import namaste_to_icd from runner

with open("namaste_data_code_mapping.json", "r", encoding="utf-8") as f:
    namaste_data_mapped = json.load(f)

EXACT_TH = 0.6 # FOR SCORE.
RELATED_TH = 0.3

NAMASTE_SYSTEM = "http://namaste.in/codes"
ICD11_SYSTEM = "http://id.who.int/icd/release/11/tm2"

def _category_for_score(score): 
    if score >= EXACT_TH:
        return "exactMatch"
    if score >= RELATED_TH:
        return "relatedMatch"
    return "noMatch"

def make_fhir_condition(
    namaste_code,       
    subject_id=None,   
    resource_id=None, 
    timestamp=None     
):
    mappings = namaste_to_icd(namaste_code)
    namaste_entry = namaste_data_mapped[namaste_code]
    namaste_display = namaste_entry['title']
    namaste_description = namaste_entry['description']

    if resource_id is None:
        resource_id = str(uuid.uuid4())

    if timestamp is None:
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    coding = []
    coding.append({
        "system": NAMASTE_SYSTEM,
        "code": namaste_code,
        "display": namaste_display
    })

    # add ICD entries as codings; include mappingCategory as an extension on each candidate
    extensions = []  # we'll also collect per-candidate extensions to attach globally if needed
    for idx, m in enumerate(mappings):
        score = float(m.get("similarity", 0.0))
        cat = _category_for_score(score)
        icd_code = m.get("icd_code")
        icd_display = m.get("title")
        coding.append({
            "system": ICD11_SYSTEM,
            "code": icd_code,
            "display": icd_display
        })
        # per-candidate extension metadata (kept outside coding because FHIR coding.extensions are per-coding --
        # we include a readable summary here and also attach a simpler global extension)
        extensions.append({
            "candidateIndex": idx,
            "system": ICD11_SYSTEM,
            "code": icd_code,
            "display": icd_display,
            "score": round(score, 4),
            "mappingCategory": cat
        })

    # Build the FHIR-like Condition resource
    condition = {
        "resourceType": "Condition",
        "id": resource_id,
        "meta": {
            "lastUpdated": timestamp
        },
        "code": {
            "coding": coding,
            # human-readable summary for quick viewing
            "text": f"{namaste_display} (mapped to ICD-11 candidates)"
        },
        "verificationStatus": "confirmed" if mappings and float(mappings[0].get("similarity", 0.0)) >= EXACT_TH else "provisional"

    }
    return json.dumps(condition, indent=2, ensure_ascii=False)

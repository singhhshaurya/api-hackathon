import json
import uuid
from datetime import datetime
from model_initialiser import mapper, namaste_data_mapped

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
    mappings = mapper.map_entry(namaste_data_mapped[namaste_code])
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
        score = m.get("similarity")
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
        "verificationStatus": "confirmed" if mappings and float(mappings[0].get("similarity", 0.0)) >= EXACT_TH else "provisional",
         "extension": [
            {
                "url": "http://namaste.in/fhir/StructureDefinition/mappingCandidates",
                "valueMappingCandidates": extensions
            }
        ]

    }
    return json.dumps(condition, indent=2, ensure_ascii=False)

'''
EXAMPLE
make_fhir_condition("AAB-30")

OUTPUT:
{
  "resourceType": "Condition",
  "id": "0c38e400-2b1d-4f5f-a94b-c3dc220ef612",
  "meta": {
    "lastUpdated": "2025-09-22T17:08:29Z"
  },
  "code": {
    "coding": [
      {
        "system": "http://namaste.in/codes",
        "code": "AAB-30",
        "display": "kesabumisputanam kevalavata"
      },
      {
        "system": "http://id.who.int/icd/release/11/tm2",
        "code": "SN4C",
        "display": "Scaling of scalp and forehead skin disorder (TM2)"
      },
      {
        "system": "http://id.who.int/icd/release/11/tm2",
        "code": "SK7Y",
        "display": "Other specified eye disorders (TM2)"
      },
      {
        "system": "http://id.who.int/icd/release/11/tm2",
        "code": "SM41",
        "display": "Jaundice disorder (TM2)"
      }
    ],
    "text": "kesabumisputanam kevalavata (mapped to ICD-11 candidates)"
  },
  "verificationStatus": "confirmed",
  "extension": [
    {
      "url": "http://namaste.in/fhir/StructureDefinition/mappingCandidates",
      "valueMappingCandidates": [
        {
          "candidateIndex": 0,
          "system": "http://id.who.int/icd/release/11/tm2",
          "code": "SN4C",
          "display": "Scaling of scalp and forehead skin disorder (TM2)",
          "score": 0.7127,
          "mappingCategory": "exactMatch"
        },
        {
          "candidateIndex": 1,
          "system": "http://id.who.int/icd/release/11/tm2",
          "code": "SK7Y",
          "display": "Other specified eye disorders (TM2)",
          "score": 0.6297,
          "mappingCategory": "exactMatch"
        },
        {
          "candidateIndex": 2,
          "system": "http://id.who.int/icd/release/11/tm2",
          "code": "SM41",
          "display": "Jaundice disorder (TM2)",
          "score": 0.6268,
          "mappingCategory": "exactMatch"
        }
      ]
    }
  ]
}

'''

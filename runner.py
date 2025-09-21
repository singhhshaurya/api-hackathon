# import mapper, namaste_data_mapped from initialiser.
import json
from datetime import date

# print(namaste_data_mapped)
concept_map_index = {}

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

def add_to_concept_map(namaste_entry, top_matches):

    EXACT_TH = 0.7
    RELATED_TH = 0.3
    def _category_for_score(score): 
        if score >= EXACT_TH:
            return "exactMatch"
        if score >= RELATED_TH:
            return "relatedMatch"
        return "noMatch"
    
    entry = {
          "code": namaste_entry['code'],
          "display": namaste_entry['title'],
          "target": []
    }
    for i in top_matches:
        p = {
            'code':i['icd_code'],
            'display':i['title'],
            "equivalence": _category_for_score(i['similarity'])
        }
        entry['target'].append(p)
    
    return entry
    



def fetch_namaste_entry(namaste_code):
    if code in namaste_data_mapped:
        return namaste_data_mapped[namaste_code]
    else:
        return None
    
# fetch_namaste_entry("AAB-51")

def namaste_to_icd(namaste_code):
    # fetch namaste entry first.
    if namaste_code in concept_map_index:
        index = concept_map_index[namaste_code]
        print("DORA")
        return concept_map['group'][0]['element'][index]
        
    namaste_entry = fetch_namaste_entry(namaste_code)
    
    if namaste_entry:
        top_matches = mapper.map_entry(namaste_entry)
        top_matches = add_to_concept_map(namaste_entry, top_matches) # gives us in conceptmap format.
        concept_map['group'][0]['element'].append(top_matches)
        concept_map_index[namaste_code] = len(concept_map['group'][0]['element']) - 1
        return top_matches # this is the ACTUAL icd codes data.
    else:
        return None


'''
EXAMPLE INPUT: namaste_to_icd('AAB-3')
EXAMPLE OUTPUT: 

{'code': 'ABB-10', 'display': 'ushmadhikyam kevalapitta', 
'target': [{'code': 'SP9Y', 'display': 'Other specified disorders affecting the whole body (TM2)', 'equivalence': 'exactMatch'}, 
           {'code': 'SM87', 'display': 'Dysuria disorder (TM2)', 'equivalence': 'relatedMatch'}, 
           {'code': 'SP5Y', 'display': 'Other specified febricity disorders (TM2)', 'equivalence': 'relatedMatch'}]
}
'''

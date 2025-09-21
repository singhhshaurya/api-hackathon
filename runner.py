final2 = []


def fetch_namaste_entry(namaste_code):
    if code in namaste_data_mapped:
        return namaste_data_mapped[namaste_code]
    else:
        return None
    
# fetch_namaste_entry("AAB-51")

def namaste_to_icd(namaste_code):
    # fetch namaste entry first.
    namaste_entry = fetch_namaste_entry(namaste_code)
    if namaste_entry:
        top_matches = mapper.map_entry(namaste_entry)
        return top_matches
    else:
        return None


'''
EXAMPLE INPUT: namaste_to_icd('AAB-3')
EXAMPLE OUTPUT: 
[{'icd_code': 'SP9Y',
  'title': 'Other specified disorders affecting the whole body (TM2)',
  'similarity': 1.0000001192092896},
 {'icd_code': 'SM1Y',
  'title': 'Other specified oral cavity disorders (TM2)',
  'similarity': 0.6810562014579773},
 {'icd_code': 'SK95',
  'title': 'Rhinitis disorder (TM2)',
  'similarity': 0.6641131639480591}]
'''

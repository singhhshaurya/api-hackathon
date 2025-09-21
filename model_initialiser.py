import json

with open("tm2_data_main.json", "r", encoding="utf-8") as f:
    icd_list = json.load(f)

with open("sanskrit-to-english.json", "r", encoding="utf-8") as f:
    alias = json.load(f)

with open("namaste_entries.json", "r", encoding="utf-8") as f:
    namaste_data_mapped = json.load(f)

#print(icd_list)
mapper = Mapper(alias, icd_list)

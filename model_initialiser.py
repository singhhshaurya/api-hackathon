import json

with open("tm2_data_main.json", "r", encoding="utf-8") as f:
    icd_list = json.load(f)

with open("sanskrit_aliases_main1.json", "r", encoding="utf-8") as f:
    alias = json.load(f)

with open("namaste_data_main2.json", "r", encoding="utf-8") as f:
    namaste_list = json.load(f)
    
#print(alias)
#print(namaste_list)

mapper = Mapper(icd_entries=icd_list, alias_dict=alias, model_name="paraphrase-multilingual-mpnet-base-v2", use_faiss=False)

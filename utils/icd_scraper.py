
import requests
import json
import urllib3
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
client_id=os.getenv("ICD_CLIENT_ID")
client_secret=os.getenv("ICD_CLIENT_SECRET")
token_endpoint = 'https://icdaccessmanagement.who.int/connect/token'
scope = 'icdapi_access'
grant_type = 'client_credentials'
urllib3.disable_warnings()

payload = {'client_id': client_id, 
	   	   'client_secret': client_secret, 
           'scope': scope, 
           'grant_type': grant_type}


r = requests.post(token_endpoint, data=payload, verify=False).json()
token = r['access_token']
print(token)


# uri = 'https://id.who.int/fhir/CodeSystem/$validate-code?url=http://id.who.int/icd/release/11/mms&code=1A00'
uri = 'https://id.who.int/icd/release/11/2025-01/mms/1147241349'

# HTTP header fields to set
headers = {'Authorization':  'Bearer '+token, 
           'Accept': 'application/json', 
           'Accept-Language': 'en',
	   'API-Version': 'v2'}


results = []
def runreq(root):
    r = requests.get(root, headers=headers, verify=False).json()
    if 'child' in r:
        for i in r['child']:
            runreq(i)
    if r['code'] == "":
        return
    synonyms = []
    # synonyms.append(r['title']['@value'])
    indexterms = r['indexTerm']
    for term in indexterms:
        terms = term['label']['@value'].split(',')
        [synonyms.append(m) for m in terms]
    val = {'code': r['code'], 'synonyms':synonyms}
    print(val)
    results.append(val)
    
    


runreq(uri)
df = pd.DataFrame(results)
df.to_csv('icd11_tm2_codes.csv', index=False, encoding='utf-8')



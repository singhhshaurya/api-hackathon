# NAMASTE–ICD Mapper 

This project builds an **NLP-based prototype** to map **Ayurvedic NAMASTE codes** to their corresponding **ICD-11 TM2 codes** using semantic similarity.  

---

## Problem  
- The **NAMASTE dataset** contains Ayurvedic disease names in Sanskrit, along with synonyms and descriptions.  
- The **ICD-11 TM2 dataset** contains Traditional Medicine extensions with standardized codes.  
- Manual mapping between them is time-consuming and inconsistent.  

---

## Solution  
We built an **NLP Mapper** that:  
1. Takes a **NAMASTE entry** (title, synonyms, description).  
2. Expands it with **custom Sanskrit → English aliases**.  
3. Generates embeddings using **Sentence Transformers** (`all-MiniLM-L6-v2`).  
4. Compares against ICD TM2 embeddings using **cosine similarity**.  
5. Returns the **Top-3 best ICD code matches** with one similarity score each.  

---

## Features  
- **Simple interface**: input one NAMASTE entry, output top 3 ICD codes.  
- **High accuracy**: uses semantic embeddings (beyond string matching).  
- **Custom alias integration**: leverages Sanskrit-to-English medical mappings.  
- **Lightweight**: works without external indexing systems (no FAISS).  

---

## Tech Stack  
- **Python 3.9+**  
- **SentenceTransformers** (`all-MiniLM-L6-v2`)  
- **Scikit-learn** (for cosine similarity)  
- **Custom preprocessing & tokenizer** for Sanskrit/English text  

---


### Example  

- **Input**: NAMASTE CODE : AAB-3
- **Output**
```python
{
  "code": "ABB-10",
  "display": "ushmadhikyam kevalapitta",
  "target": [
    {
      "code": "SP9Y",
      "display": "Other specified disorders affecting the whole body (TM2)",
      "equivalence": "exactMatch"
    },
    {
      "code": "SM87",
      "display": "Dysuria disorder (TM2)",
      "equivalence": "relatedMatch"
    },
    {
      "code": "SP5Y",
      "display": "Other specified febricity disorders (TM2)",
      "equivalence": "relatedMatch"
    }
  ]
}
  

```
## Future Improvements  

- Optimize embeddings with **domain-specific fine-tuning**  
- Handle **multi-lingual Sanskrit transliterations** better  
- Build a **UI dashboard** for quick mappings  

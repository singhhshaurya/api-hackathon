final2 = []

ctr = 0
for namaste_entry in namaste_list:
    top_matches = mapper.map_entry(namaste_entry)
    final2.append({namaste_entry['title']:top_matches})
    print(ctr, 'done')
    ctr += 1

# from model import namaste_list
final = []

ctr = 0
for nam in namaste_list:
    ctr += 1
    res = mapper.map_single(nam)
    final.append({res['namaste_title']:[]}) 
    for i in range(len(res['top_k'])):
        final[-1][res['namaste_title']].append([res['top_k'][i]['code'], res['top_k'][i]['title'], res['top_k'][i]['sim_agg']])
    print(ctr, 'done')

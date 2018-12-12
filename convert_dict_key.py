import json

with open('submit.json') as f:
    lis = json.load(f)

a = []
for l in lis: 
    b={}
    b['image_id'] = l['disease_class']
    b['disease_class'] = l['image_id']
    a.append(b)

with open('submits.json', 'w') as ff:
    json.dump(a,ff)
print('finish...')
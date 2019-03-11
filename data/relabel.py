import os,sys,random

D = [line.strip() for line in open("bd_master.txt", 'r')]

labels = set()

T = open("../splits/bd.labeled.train",'w')
E = open("../splits/bd.labeled.test",'w')
V = open("../splits/bd.labeled.val",'w')
data = {}
for i in range(int(len(D)/2)):
  code = D[2*i].replace(">","").split("_")[0]
  if len(code) > 10:
    print(code)
    print(i)
    sys.exit()
  seq = D[2*i + 1]
  if code not in data:
    data[code] = set()
  data[code].add(seq)
  labels.add(code)

for code in data:
  for seq in data[code]:
    r = random.random()
    if r < 0.7:
      T.write("{} {}\n".format(code, seq))
    elif r < 0.9:
      E.write("{} {}\n".format(code, seq))
    else:
      V.write("{} {}\n".format(code, seq))

T.close()
E.close()
V.close()

L = open("../splits/bd.labels.txt",'w')
for label in labels:
  L.write("{}\n".format(label))
L.close()

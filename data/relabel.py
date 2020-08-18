import os,sys,random

# Pass in master database e.g. "bd_master.txt"
if len(sys.argv) == 1:
  print("Invalid usage, please pass database file (e.g. bd_master.txt)")
  sys.exit()

D = [line.strip() for line in open(sys.argv[1], 'r')]

labels = set()

T = open("../data/custom.labeled.train",'w')
E = open("../data/custom.labeled.test",'w')
V = open("../data/custom.labeled.val",'w')
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

L = open("../data/custom.labels.txt",'w')
for label in labels:
  L.write("{}\n".format(label))
L.close()

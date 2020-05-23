#Label HCO sequence

labels=['A','B','C','D','E','F','G','H','I','J','K','CNOR','QNOR','BNOR','ENOR','GNOR','NNOR','SNOR','NOD']

#Import sequences
outfile1=open('HCO_renamed.txt','w')
outfile2=open('HCO_labels.txt','w')
for label in labels:
	file='%s'%label + '_sequences_unique.txt'
	with open(file) as fp:
		sequences = fp.readlines()
	for i in range(0,len(sequences)-1,2):
		name=sequences[i]
		seq=sequences[i+1]
		new_name='%s'%label+'_%s'%int(0.5*i+1)
		outfile1.write('>'+new_name+'\n'+seq)
		outfile2.write(name.rstrip()+'\t'+new_name+'\n')

outfile1.close()
outfile2.close()
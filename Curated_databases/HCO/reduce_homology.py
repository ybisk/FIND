import os, sys, subprocess
labels=['A','B','C','D','E','F','G','H','I','J','K','BNOR','CNOR','ENOR','GNOR','NNOR','SNOR','QNOR','NOD','HCO']
cutoff='0.4'
dir='Filtered_%s'%cutoff
os.mkdir(dir)
for label in labels:
	infile='%s'%label+'_sequences_unique.txt'
	outfile='%s'%label+'_%s'%cutoff+'_sequences.txt'
	subprocess.run(["usearch","-cluster_fast", '%s'%infile, "-id",'%s'%cutoff,"-centroids",'Filtered_0.4/%s'%outfile])

os.chdir(dir)

countfile='HCO_count_'+'%s'%cutoff+'.txt'
outfile1=open(countfile,'w')
#Reformat_the_files
for label in labels:
	sequence_dict={}
	file='%s'%label+'_%s'%cutoff+'_sequences.txt'
	with open(file) as fp:
		sequences=fp.readlines()
	for i in range(len(sequences)-1):
		if sequences[i].find('>')!=-1:
			j=i
			i=i+1
			while (sequences[i].find('>')==-1 and i<(len(sequences)-1)):
				i=i+1
			if i<(len(sequences)-1):
				string=''.join(sequences[j+1:i])
			elif i==len(sequences)-2:
				string=''.join(sequences[j+1:i+1])
		sequence_dict[sequences[j]]=string.replace("\n","")
	file_out='%s'%label+'_%s'%cutoff+'_sequences_formatted.txt'
	outfile2=open(file_out, "w")
	for key,val in sequence_dict.items():
		outfile2.write(key+val+'\n')
	n=len(sequence_dict.keys())
	outfile1.write(label+'\t'+str(n)+'\n')
	outfile2.close()
outfile1.close()

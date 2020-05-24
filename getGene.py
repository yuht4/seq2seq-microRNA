import RNA
import requests, sys
import sqlite3
import os
from Bio.Blast.Applications import NcbiblastnCommandline


from Yu import translate
from Yu import getModel


#mirna = '>'+'refseq_1'+'\n'+'UCACCAGCCCUGUGUUCCCUAG'
def Blast_seq(mirna):

    with open('./data/mirna.fasta','w+') as f:
        f.write('>'+'refseq_1'+'\n'+ str(mirna))
    if os.path.isfile('blast_result.csv'):   
        os.remove('blast_result.csv')    
    blastx_cline = NcbiblastnCommandline(query='./data/mirna.fasta', db="./data/human_mirna", evalue=0.05,outfmt=10, out="blast_result.csv",word_size= 7, gapopen = 50, gapextend = 3, strand= 'both')
    stdout, stderr = blastx_cline()
    list_of_mirna = []
    try: 
        with open('blast_result.csv','r+') as f:
            lines = f.read()
            if '\n' in lines:
                lines = lines.split('\n')
               
            for line in lines:
                if ',' in line:
                    list_of_mirna.append(line.split(',')[1])
        if len(list_of_mirna)>0:
            
            return list_of_mirna        
        else:
            return None
    except:
        return None



def get_3utr(transcript_id):
    link = ('https://asia.ensembl.org/Homo_sapiens/Export/Output/Transcript?db=core;'+
    'flank3_display=0;flank5_display=0;output=fasta;strand=feature;'+
    't={}'.format(transcript_id)+';param=utr3;genomic=unmasked;_format=Text')
    utr = requests.get(link)
    
    
       
    utr = utr.text.split('>')[1]
    utr_split = utr.split('\n')
    utr_seq=''
    #print('start_for') 
    if 'utr3'in utr_split[0]:
        for fasta in utr_split[1:]:
            utr_seq=utr_seq+fasta.replace('\n','')
            
        return(utr_seq)    
    else:
        return(None)



def get_seq_to_predict(GeneSymbol):

    conn = sqlite3.connect('./data/linker.db')
    c = conn.cursor()
    c.execute("SELECT * FROM link WHERE Gene_symbol=:Gene_symbol",{'Gene_symbol':GeneSymbol})

    GeneSymbol = GeneSymbol.upper()
    seq_to_predict=[]
    
    linker = c.fetchall()
   
    
    if linker is None:
         print("no match gene found")
        
    else:
        print('No. of protein coding transcript found = '+str(len(linker)))
        count = 0
        for line in linker:
         
            count = count+1
            print('Analysing tanscript ' +str(count)+'......')
            
            id_e = line[2]
            print('Transcript ID = '+str(id_e))
            server = "https://rest.ensembl.org"
            ext = "/sequence/id/{}?".format(id_e.split('.')[0])
             
            r = requests.get(server+ext, headers={ "Content-Type" : "text/plain","Connection": "close"})
             
            if not r.ok:
              r.raise_for_status()
              sys.exit()
            print('Calculating binding sites')
            #s.keep_alive = False
            
            utr_seq=get_3utr(id_e)
            #print('start_for') 
            if utr_seq is not None:
              
                seq=r.text.replace('T','U')# l in negative(-l)
                print('Length of transcript = '+str(len(seq)))
                print("Lenght of 3'UTR = "+str(len(utr_seq))+'\n'+'\n')
                
                
                if len(seq) > len(utr_seq)*1.5:
                    l=len(seq)-int(len(utr_seq)*1.5)
                else:
                    l= len(seq)-len(utr_seq)
                
                
                # compute minimum free energy (MFE) and corresponding structure
                
    
                n= RNA.pfl_fold_up(seq,16,40,80)
                for i in range(l,len(seq)):
                    if n[i][4]>0.2:                  
                     
                        s = seq[i-23:i]
                        seq_to_predict.append(s[::-1])
            else:
                continue
        return list(set(seq_to_predict))
                
                
                
                
        
def main(GeneSymbol = 'TNF'):

    model = getModel();
    model.load_weights('transformer_embed.h5')

    GeneSymbol = GeneSymbol.upper()
    print("Gene Symbol: {}".format(GeneSymbol))

    list_mRNA = get_seq_to_predict(GeneSymbol)

    print(str(len(list_mRNA)) + " mRNA segments were found\n")    
    print('Predicting miRNA sequences.......\n')

    prediction_mRNA_microRNA = []

    for mRNA in list_mRNA:

        microRNA = translate(mRNA, model)
 
        if RNA.fold(mRNA[::-1][:] + 'LLLLLLLL' + microRNA[:])[1] <-8:

            prediction_mRNA_microRNA.append([mRNA, microRNA])


    mirna_prediction = []

    for mRNA, mirna in prediction_mRNA_microRNA:
        mirna_name = Blast_seq(mirna)

        if mirna_name != None:
            for name in mirna_name:
                mirna_prediction.append(name)
        else:
            mirna_prediction.append(mirna)

    mirna_prediction = list(set(mirna_prediction))

    novel = []
    known = []

    for mi in mirna_prediction:
        if 'hsa' in mi:
            known.append(mi)
        else:
            novel.append(mi)

    print('Total miRNA predicted: ' + str(len(mirna_prediction)) + '\n')
    print('Total predicted mirna which are present in miRbase v22: ' + str(len(known)) + '\n')
    print('Total Predicted novel mirna which are not present in miRbase v22: ' + str(len(novel)) + '\n')

    Path = './results/' + GeneSymbol + '.txt'

    if os.path.isfile(Path):
        os.remove(Path)

    with open(Path, 'w') as f:
        f.write('Gene Symbol: ' + GeneSymbol + '\n' + 'Predicted mirna which are present in miRbase v22:' + '\n')
        for name in known:
            f.write(name + '\n')

        f.write('\n\n\n' + 'Predicted novel mirna which are not present in miRbase v22:' + '\n')
        for seq in novel:
            f.write(seq + '\n')



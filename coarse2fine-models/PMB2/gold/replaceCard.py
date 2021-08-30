from __future__ import unicode_literals, print_function, division
from io import open
import re
import argparse


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-src', type=str, default='train_input.txtRaw')
parser.add_argument('-trg', type=str, default='train_input.txt')
args = parser.parse_args()
rel=[]
with open(args.src, 'r', encoding='utf-8') as r, open(args.trg, 'w', encoding='utf-8') as w:
    while True:
        l1 = r.readline()
        l2 = r.readline()
        words=l2.strip().split()
        l3 = r.readline()
        l4 = r.readline()
        l5 = r.readline()
        l6 = r.readline().strip()
        tokens=l6.strip().split()
        l7 = r.readline()
        if l1 == "":
            break
        w.write(l1)
        w.write(l2)
        w.write(l3)
        w.write(l4)
        w.write(l5)
        for i in range(len(tokens)-1):
            if tokens[i]=='RESULT(' and tokens[i+1]=='DRS(':
                tokens[i] = 'RESULTS('
            if tokens[i][-1]=='(' and tokens[i+1][-1].isdigit(): # and not tokens[i][:-1] in l2:
                if re.match(r'^\d\d:\d\d$',tokens[i][:-1]):
                    tokens[i]='CLOCK_TIME('
                if re.match(r'^\d+(\.\d+)?$',tokens[i][:-1]) or re.match(r'^\d+X$',tokens[i][:-1]) or re.match(r'^\d+-\d+$',tokens[i][:-1]):
                    tokens[i]='CARD_REL('
                if tokens[i]=='fed~up(':
                    tokens[i]='feed~up('
                if tokens[i]=='used~to(':
                    tokens[i]='use~to('
                if tokens[i]=='fed~up(':
                    tokens[i]='feed~up('
                if tokens[i]=='waiting~room(':
                    tokens[i]='wait~room('
                if tokens[i]=="cote~d'ivoire(":
                    tokens[i]="côte~d'ivoire("
                if tokens[i]=="andris~berzins(":
                    tokens[i]="andris~bērziņš("
                if tokens[i].isupper() and '~' in tokens[i]:
                    tokens[i]=tokens[i].replace("~", "-")
                if tokens[i]=='PHONE~NUMBER(':
                    tokens[i]='PHONE-NUMBER('
        x=1
        e=1
        s=1
        t=1
        t10=0
        x10=0
        dictR={}
        for i in range(len(tokens)):
            if tokens[i][-1].isdigit() and not tokens[i][0]=='P':
                if tokens[i] in dictR:
                    pass
                else:
                    if tokens[i].startswith('T10'):
                        dictR[tokens[i]]='Y'+str(t10)
                        t10+=1
                    elif tokens[i].startswith('X10'):
                        dictR[tokens[i]]='Z'+str(x10)
                        x10+=1
                    elif tokens[i].startswith('X'):
                        dictR[tokens[i]]='X'+str(x)
                        x+=1
                    elif tokens[i].startswith('E'):
                        dictR[tokens[i]]='E'+str(e)
                        e+=1
                    elif tokens[i].startswith('S'):
                        dictR[tokens[i]]='S'+str(s)
                        s+=1
                    elif tokens[i].startswith('T'):
                        dictR[tokens[i]]='T'+str(t)
                        t+=1
                    else:
                        print(tokens[i])
                        assert False
                tokens[i] = dictR[tokens[i]]

        w.write(' '.join(tokens)+'\n')
        w.write(l7)

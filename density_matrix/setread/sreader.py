#!/usr/bin/python
''' This file is used to read my python measurement file and extract its
measurement settings and parameters, takes the fist defined values it finds.'''

import numpy as np
import re
import sys

# searchstring = sys.argv[1]
filename = '1150__Gsweep.py'  # sys.argv[1]

def maketable(rows, cols):
    ''' to avoid Python creating a 2d-list out of aliases '''
    mytable = []
    for row in range(rows):
        mytable += [cols*[None]]
    return mytable


reg_dic = {}  # Dictionary for regrex
reg_dic['read_number'] = r'[-\.\de]*'
reg_dic['find_value'] = r"\s*=\s*(['\s\w\\/\(\).e\d\*-]*)"
reg_dic['find_inst'] = r"\n(\w+)*\s*=\s*[\w']+\(([\w\s'=]+,.*?=[^#\s].*?)\)[\r|\n]"


# Find all the basic values defined in a single line
st = 4*[None]
st[0] = 'BW'
st[1] = 'lsamples'
st[2] = 'f1'
st[3] = 'f2'
values = 4*[None]
with open(filename, 'r') as a:
    for i in a.readlines():
        for num in range(len(st)):
            m = re.search('^\s*'+st[num]+'\s*=\s*([-\.\de]*)', i)
            if m:
                values[num] = eval(m.group(1))


with open(filename, 'r') as a:
    b = a.read()


subfolder = re.search("fol\w*\s*=\s*'(.*?)'", b).groups()[0]
path = r'\\' + subfolder

myinst = re.findall(reg_dic['find_inst'], b, flags=re.MULTILINE | re.DOTALL)
inst_table = maketable(len(myinst)+1, 4)
inst_table[0] = ['Instrument', 'start', 'stop', 'pt']
for num, string in enumerate(myinst):
    inst_table[num+1][0] = string[0]
    inst_table[num+1][1] = re.search(r'start'+reg_dic['find_value'], string[1]).groups()[0]
    inst_table[num+1][2] = re.search(r'stop'+reg_dic['find_value'], string[1]).groups()[0]
    inst_table[num+1][3] = re.search(r'pt'+reg_dic['find_value'], string[1]).groups()[0]

print path
for string in st, values: print string
for inst in inst_table: print inst

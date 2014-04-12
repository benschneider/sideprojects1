import numpy as np
from struct import pack, unpack

def loadmtx(input):
    '''MTX - file parser by Ben Schneider
     input = 'filename.mtx'
     output = [matrix, header]
     (access with:  
     output[0]  = header
     output[1]  = Matrix
     output[1][z][y][x] )
    '''
    f = open(input,'rb')

    line = f.readline()
    header = line[:-1].split(',')
    #header = line
    
    line = f.readline()    
    a= line[:-1].split(' ')
    s = np.array(map(float, a))

    raw = f.read() #reads everything else
    f.close()
    if (s[3] == 4):
        data = unpack('f'*(s[2]*s[1]*s[0]),raw) #uses float
        M = np.reshape(data, (s[2],s[1],s[0]),order="F")
    else:
        data = unpack('d'*(s[2]*s[1]*s[0]),raw) #uses double
        M = np.reshape(data, (s[2],s[1],s[0]),order="F")        
    #output = [header,M]; 
    #del header,M,data
    return M, header#output

#note reshape modes:
#a
#Out[133]: 
# array([[1, 2, 3],
#	[4, 5, 6]])
#
#In [134]: a.reshape(3,2, order='F')
#Out[134]: 
# array([[1, 5],
#	[4, 3],
#	[2, 6]])
#
#In [135]: a.reshape(3,2, order='c')
#Out[135]: 
# array([[1, 2],
#	[3, 4],
#	[5, 6]])
# -B
    
def savemtx(filename,data):
    '''MTX - file parser by Ben Schneider    
     input = 'filename.mtx',[header,matrix]
    '''
    f = open(filename,'wb')
    
    header = ",".join(data[0])
    f.write(header +'\n')    
    
    s = len(data[1][0][0]),len(data[1][0]),len(data[1]),8 #(x ,y ,z , 8)
    line = " ".join(str(b) for b in s) #'x y z 8'
    f.write(line +'\n')  #'x y z 8 \n'

    M = data[1]
    if  (len(s)==4):
        raw2 = np.reshape(M,s[2]*s[1]*s[0],order="F")
        raw = pack('%sd' % len(raw2), *raw2)
        f.write(raw);

    f.close()

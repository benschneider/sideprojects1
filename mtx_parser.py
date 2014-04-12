'''
MTX - file parser by Ben Schneider

for now you can load it with 'execfile('mtx_parser.py)'
it will add the following content.

content:
    loaddat : load an ASCII data file ( loaddat('file.dat') )
    savedat : save an ASCII data file ( savedat('file.dat') )
    loadmtx : load a binary data file ( loadmtx('file.mtx') )
    savemtx : save a binary data file ( savemtx('file.mtx', 3d_numpy_array))

missing:
-   requires a default header when saving MTX
-   additional descriptions
-   Change into an importable thingy

'''
import numpy as np
from struct import pack, unpack

def loaddat(*inputs):
    '''
    This simply uses the numpy.genfromtxt function to
    load a data containing file in ascii
    (It rotates the output such that each colum can be accessed easily)

    example:
    in the directory:
    1.dat:
        1   2   a
        3   b   4
        c   5   6
        7   8   d

    >> A = loaddat('1.dat')
    >> A[0]
    (1,3,c,7)
    '''
    file_data = np.genfromtxt(*inputs)
    outputs = zip(*file_data)
    return outputs

def savedat(*args):
    '''filename, data, arguments'''
    #args[1] = array(*args[1])
    np.savetxt(*args, delimiter=',')

def loadmtx(filename):
    '''
     output = matrix, header
     (access with:
     output[0]  = header
     output[1]  = Matrix
     output[1][z][y][x] )

     i.e.: mtx, header = loadmtx('filename.mtx')
    '''
    f = open(filename, 'rb')

    line = f.readline()
    header = line[:-1].split(',')
    #header = line

    line = f.readline()
    a = line[:-1].split(' ')
    s = np.array(map(float, a))

    raw = f.read() #reads everything else
    f.close()
    if s[3] == 4:
        data = unpack('f'*(s[2]*s[1]*s[0]), raw) #uses float
        M = np.reshape(data, (s[2], s[1], s[0]), order="F")
    else:
        data = unpack('d'*(s[2]*s[1]*s[0]), raw) #uses double
        M = np.reshape(data, (s[2], s[1], s[0]), order="F")
    return M, header

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

def savemtx(filename, *data):
    '''MTX - file parser by Ben Schneider
     input = 'filename.mtx',[header,matrix]
    '''
    f = open(filename, 'wb')

    #if len(data[1]) == : #check if header was given
    header = ",".join(data[1]) #write the header in the first line
    f.write(header +'\n')

    s = len(data[0][0][0]), len(data[0][0]), len(data[0]), 8 #(x ,y ,z , 8)
    line = " ".join(str(b) for b in s) #'x y z 8'
    f.write(line +'\n')  #'x y z 8 \n'

    M = data[0]
    if  len(s) == 4:
        raw2 = np.reshape(M, s[2]*s[1]*s[0], order="F")
        raw = pack('%sd' % len(raw2), *raw2)
        f.write(raw)

    f.close()

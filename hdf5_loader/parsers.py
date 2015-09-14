'''
MTX - file parser

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

- B
'''
import numpy as np
from struct import pack, unpack
import csv
from os import path
import sys
from shutil import copy
import h5py


def load_hdf5(filename):
    '''
    Loads an HDF5 file which was created by Lab_Control soft.
    This simply uses the h5py module to load hdf5 files.
    '''
    hdf5data = emptyClass()
    with h5py.File(filename, 'r') as f:
        hdf5data.data = np.array(f['Data']['Data'])
        hdf5data.channel = np.array(f['Data']['Channel names'])

    create_hdf5_dims_2d(hdf5data)  # add axes, and labels
    hdf5data.shape = hdf5data.data[:, 1, :].shape

    return hdf5data


def create_hdf5_dims_2d(hdf5data):
    '''
    Currently under construction!
    requires hdf5data set and selected channel number 'cname'
    returns an the selected data set and
    objects of the cor. dimensions. (name, limits, points..)
    np. array shape is as (n1, n2, n3)
    '''
    def _create_dim(d1, num):
        dim_ax = dim(name=hdf5data.channel[num][0],
                     start=d1[0],
                     stop=d1[-1],
                     pt=len(d1),
                     scale=1)
        return dim_ax

    # lab-control soft changes the data in a strange way:
    # dim 1 <- channel 2
    # dim_2 <- channel 1
    # dim_3 <- channel 3

    d0 = hdf5data.data[:, 0, 0]
    d1 = hdf5data.data[0, 1, :]
    d2 = hdf5data.data[0, 2, :]

    hdf5data.n2 = _create_dim(d0, 0)
    hdf5data.n1 = _create_dim(d1, 1)
    hdf5data.n3 = _create_dim(d2, 2)


def laad_hdf5_dims_old(hdf5data):
    '''
    Currently under construction!
    requires manual addustments
    '''
    hdf5data.d1 = hdf5data.data[:, 0, 0]
    hdf5data.d2 = hdf5data.data[0, 1, :]
    hdf5data.d3 = hdf5data.data[0, 2, :]

    hdf5data.dim_2 = dim(name=hdf5data.channel[0][0],
                         start=hdf5data.d1[0],
                         stop=hdf5data.d1[-1],
                         pt=len(hdf5data.d1),
                         scale=1)

    hdf5data.dim_3 = dim(name=hdf5data.channel[1][0],
                         start=hdf5data.d2[0],
                         stop=hdf5data.d2[-1],
                         pt=len(hdf5data.d2),
                         scale=1)

    hdf5data.dim_1 = dim(name=hdf5data.channel[2][0],
                         start=hdf5data.d3[0],
                         stop=hdf5data.d3[-1],
                         pt=len(hdf5data.d3),
                         scale=1)


def ask_overwrite(filename):
    if path.isfile(filename):
        print 'Overwrite File? type:yes'
        a0 = raw_input()
        if a0 != 'yes':
            return sys.exit("Abort")


def copy_file(thisfile, file_add, folder=''):
    ''' folder = "somefolder\\"
    i.e.
    thisfile = '__filename__'
    copy_file(thisfile, 'bla','data\\')
    '''
    # drive = os.getcwd()                # D:\
    # filen = path.basename(thisfile)     # something.py
    ffile = path.abspath(thisfile)     # D:\something.py
    ffolder = path.dirname(thisfile)    # EMPTY
    new_ffile = (ffolder +
                 folder +
                 thisfile[:-3] +
                 '_' +
                 file_add +
                 thisfile[-3:])
    copy(ffile, new_ffile)


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


def savedat(filename1, data1, **quarks):
    '''
    redundand function
    it rotates the data nicely for gnuplot and then
    simply uses : np.savetxt(filename, data, delimiter = '\t')

    filename, data, arguments
    simply uses numpy.savetext with a
    delimiter = '\t'

    np.savetxt("QsQr.dat",stuff, delimiter ='\t')
    default: delimiter = '\t'  (works best with gnuplot even with excel)
    '''
    data1 = zip(*data1)
    print filename1
    if 'delimiter' in quarks:
        np.savetxt(filename1, data1, **quarks)
    else:
        np.savetxt(filename1, data1, delimiter='\t', **quarks)


def loadcsv(filename, delim=';'):
    # open file (using with to make sure file is closed afer use)
    with open(filename, 'Ur') as f:
        # collect tuples as a list in data,
        # then convert to an np.array and return
        data = list(tuple(rec) for rec in csv.reader(f, delimiter=delim))
        data = np.array(data, dtype=float)
    return data.transpose()


def loadmtx(filename):
    '''
    Loads an mtx file (binary compressed file)
    (first two lines of the MTX contain information of the data shape and
    what units, limits are present)
    i.e.:

    mtx, header = loadmtx('filename.mtx')

    mtx     :   will contain a 3d numpy array of the data
    header  :   will contain information on the labels and limits
    '''
    with open(filename, 'rb') as f:

        line = f.readline()
        header = line[:-1].split(',')
        # header = line

        line = f.readline()
        a = line[:-1].split(' ')
        s = np.array(map(float, a))

        raw = f.read()  # reads everything else
        f.close()

    if s[3] == 4:
        data = unpack('f'*(s[2]*s[1]*s[0]), raw)  # uses float
        M = np.reshape(data, (s[2], s[1], s[0]), order="F")
    else:
        data = unpack('d'*(s[2]*s[1]*s[0]), raw)  # uses double
        M = np.reshape(data, (s[2], s[1], s[0]), order="F")
    return M, header


# note: reshape modes
# a
# Out[133]:
#  array([[1, 2, 3],
# 	[4, 5, 6]])
#
# In [134]: a.reshape(3,2, order='F')
# Out[134]:
#  array([[1, 5],
# 	[4, 3],
# 	[2, 6]])
#
# In [135]: a.reshape(3,2, order='c')
# Out[135]:
#  array([[1, 2],
# 	[3, 4],
# 	[5, 6]])
# def test1(*test1,**test2):
#     '''A function to test arcs and quarks in python'''
#     if 'head' in test2:
#         return test2
#     else:
#         return test1


def savemtx(filename, data, header='Units,ufo,d1,0,1,d2,0,1,d3,0,1'):
    '''MTX - file parser by Ben Schneider
    stores to the file:
    Units, Dataset name, xname, xmin, xmax,
            yname, ymin, ymax,
            zname, zmin, zmax
    nx ny nz length
    [binary data....]

    the first line is the header i.e. with
    myheader = ('Units, S11,
                Magnet (T), -1, 1,
                Volt (V), -10, 10,
                Freqeuency (Hz), 1, 10')
    savemtx('myfile.mtx',my-3d-np-array, header = myheader)
    '''
    with open(filename, 'wb') as f:
        f.write(header + '\n')

        mtxshape = data.shape
        line = (str(mtxshape[2]) +
                ' ' + str(mtxshape[1]) +
                ' ' + str(mtxshape[0]) +
                ' ' + '8')
        f.write(line+'\n')   # 'x y z 8 \n'

        raw2 = np.reshape(data, mtxshape[0]*mtxshape[1]*mtxshape[2], order="F")
        raw = pack('%sd' % len(raw2), *raw2)
        f.write(raw)
        f.close()


def make_header(dim_1, dim_2, dim_3, meas_data='(a.u)'):
    '''
    def your sweep axis/name, start and stop
    values = Measured Voltage (V)
    dim_1.name = Current (A)
    dim_1.start = 0
    dim_1.stop = 1
    dim_2.name = Voltage (V)
    ...
    dim_3.name = RF Power (dB)
    returns a text string used as 1st line of an mtx file
    '''
    header = ('Units,' + meas_data + ',' +
              dim_1.name + ',' + str(dim_1.start) + ',' + str(dim_1.stop)
              + ',' +
              dim_2.name + ',' + str(dim_2.start) + ',' + str(dim_2.stop)
              + ',' +
              dim_3.name + ',' + str(dim_3.start) + ',' + str(dim_3.stop))
    return header


def read_header(head, **quark):
    ''' reads header file and returns dim names, axis
    dim1,dim2,dim3,dim_z = read_header(head, Data=datafile)
    Data is optional
    '''
    d1 = 1
    d2 = 1
    d3 = 1
    if 'Data' in quark:
        dd = quark['Data']
        d1 = dd.shape[2]
        d2 = dd.shape[1]
        d3 = dd.shape[0]

    dim_1 = dim(name=head[2], start=eval(head[3]), stop=eval(head[4]), pt=d1)
    dim_2 = dim(name=head[5], start=eval(head[6]), stop=eval(head[7]), pt=d2)
    dim_3 = dim(name=head[8], start=eval(head[9]), stop=eval(head[10]), pt=d3)
    dim_z = dim(name=head[1])
    return dim_1, dim_2, dim_3, dim_z


def read_header_old(head, **quark):
    ''' reads header file and returns dim names, axis
    dim1,dim2,dim3,dim_z = read_header(head, Data=datafile)
    Data is optional
    '''
    d1 = 1
    d2 = 1
    d3 = 1
    if 'Data' in quark:
        dd = quark['Data']
        d1 = dd.shape[2]
        d2 = dd.shape[1]
        d3 = dd.shape[0]

    dim_1 = dim(name=head[2], start=eval(head[3]), stop=eval(head[4]), pt=d1)
    dim_2 = dim(name=head[5], start=eval(head[7]), stop=eval(head[6]), pt=d2)
    dim_3 = dim(name=head[8], start=eval(head[9]), stop=eval(head[10]), pt=d3)
    dim_z = dim(name=head[1])
    return dim_1, dim_2, dim_3, dim_z


class dim():
    def __init__(self, name='ufo', start=0, stop=0, pt=1, scale=1):
        self.name = name   # ufo: unknown fried object
        self.start = start
        self.stop = stop
        self.pt = pt
        self.lin = np.linspace(self.start, self.stop, self.pt)*scale
        self.scale = scale

    def update_lin(self):
        self.lin = np.linspace(self.start, self.stop, self.pt)*self.scale


class emptyClass():
    '''
    Just an empty class, that can be used for storage
    '''
    def __init__(self):
        pass

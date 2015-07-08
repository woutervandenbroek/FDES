# ====================================================================
#
# Copyright (C) 2015 Wouter Van den Broek, Xiaoming Jiang
# 
# This file is part of FDES.
# 
# FDES is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# FDES is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FDES. If not, see <http://www.gnu.org/licenses/>.
# 
# Email: wouter.vandenbroek@uni-ulm.de, wouter.vandenbroek1@gmail.com,
#        xiaoming.jiang@uni-ulm.de, jiang.xiaoming1984@gmail.com
# 
# ====================================================================


import numpy as np
import ctypes
import scanf
from ctypes import *

def cuda_FDES(gpu_Index, print_Level, input_name, image_name, emd_save_name, pointerAtomsArray, numAtoms, pointerImagesArray):
    FDES_dll = ctypes.CDLL('FDES_SHARED_LIB.dll', mode=ctypes.RTLD_GLOBAL)
    func = FDES_dll.FDES
    return func(gpu_Index, print_Level, input_name, image_name, emd_save_name, pointerAtomsArray, numAtoms, pointerImagesArray)

f = open('dataFDES.cnf', 'r')
contents = f.read()
f.close()

nAt = 0
for row in contents.split("\n"):
  if 'atom:' in row:
    nAt = nAt + 1

s = 'Numer of atoms ' + repr(nAt)
print s  


for row in contents.split("\n"):
  if 'image_size_x:' in row:
    scanfTemp = scanf.sscanf(row, 'image_size_x: %d' )
    nx = scanfTemp[0]
  if 'image_size_y:' in row:
    scanfTemp = scanf.sscanf(row, 'image_size_y: %d' )
    ny = scanfTemp[0]
  if 'image_size_z:' in row:
    scanfTemp = scanf.sscanf(row, 'image_size_z: %d' )
    nz = scanfTemp[0]

s = 'Image size ' + repr(nx) + ' ' +  repr(ny) + ' ' + repr(nz)
print s 

atomsArray = np.zeros(nAt* 6)
atomsArray = atomsArray.astype(np.float32)
atomsArray_p = atomsArray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

imagesArray = np.zeros(nx*ny*nz)
imagesArray = imagesArray.astype(np.float32)
imagesArray_p = imagesArray.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

idex = 0
for row in contents.split("\n"):
  if 'atom:' in row:
   line = scanf.sscanf(row, 'atom: %f  %f %f %f %f %f' )
   atomsArray[idex*6+0] = line[0]
   atomsArray[idex*6+1] = line[1]
   atomsArray[idex*6+2] = line[2]
   atomsArray[idex*6+3] = line[3]
   atomsArray[idex*6+4] = line[4]
   atomsArray[idex*6+5] = line[5]
   idex = idex + 1
   
f = open('dataFDESRead.txt', 'w')
for x in xrange(0, nAt):
  buf = 'atom: %d %14.8g %14.8g %14.8g %14.8g %d \n' % (atomsArray[x*6+0], atomsArray[x*6+1],atomsArray[x*6+2], atomsArray[x*6+3],atomsArray[x*6+4], atomsArray[x*6+5])
  f.write(buf)

f.close()




cuda_FDES(0, 0, 'dataFDES.cnf', 'Measurements.bin', 'results.emd',atomsArray_p,nAt, imagesArray_p)


imagesArray.astype('float32').tofile('test.raw')




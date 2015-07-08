/*==================================================================

Copyright (C) 2015 Wouter Van den Broek, Xiaoming Jiang

This file is part of FDES.

FDES is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FDES is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FDES. If not, see <http://www.gnu.org/licenses/>.

Email: wouter.vandenbroek@uni-ulm.de, wouter.vandenbroek1@gmail.com,
       xiaoming.jiang@uni-ulm.de, jiang.xiaoming1984@gmail.com 

===================================================================*/

#ifndef coordArithmetic_kuzgqwefgajsdhbflquihrtqwejrohnag
#define coordArithmetic_kuzgqwefgajsdhbflquihrtqwejrohnag

#define iwCoord(s,i,m) if ((i) > (m)/2) {(s) = (i) - (m);} else {(s) = (i);}

#define owCoord(t,i,m) ((t) = ((i) - ((m)/2))) //m >> 1;

#define iwCoordIp(i,m) if ((i) > (m)/2) {(i) -= (m);}

#define owCoordIp(i,m) ((i) -= ((m)/2))

#define dbCoord(i1, i2, j, m1) ((i1) = ((j) % (m1))); ((i2) = ((j) - (i1)) /(m1))

#define trCoord( i1, i2, i3, j, m1, m2 ) ( (i1) = ( ( (j) % ( (m1) * (m2) ) ) % (m1) ) ); ( (i2) = ( ( ( (j) % ( (m1) * (m2) ) ) - (i1) ) / (m1) ) ); ( (i3) = ( ( (j) - (i1) - ( (m1) * (i2) ) ) / ( (m1) * (m2) ) ) )

#define sgCoord(j, i1, i2, m1) ((j) = (((i2) * (m1)) + (i1)))

#define sgCoord3D(j, i1, i2, i3, m1, m2) ( (j) = ( ( (m1)*(m2)*(i3) ) + ( (m1)*(i2) ) + (i1) ) )

#endif
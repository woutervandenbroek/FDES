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

#include <stdio.h>
#include <stdlib.h>
#include"paramStructure.h"

void readBinary ( const char* filename, float* I, int size )
{
    FILE* fr;
    fr = fopen ( filename , "rb" );

    size_t nread;
    nread = fread ( ( void* ) I, sizeof ( float ), size, fr );
    if(nread != (unsigned int) size )
    {
      exit(0);
    }
    fclose ( fr );
}

void writeBinary (const char* filename, float* f, int size )
{
    FILE* fw;
    fw = fopen ( filename, "wb" );

    fwrite ( ( const void* ) f, sizeof ( float ), size, fw );
    fclose ( fw );
}

void appendBinary (const char* filename, float* f, int size )
{
    FILE* fw;
    fw = fopen ( filename, "ab" );

    fwrite ( ( const void* ) f, sizeof ( float ), size, fw );
    fclose ( fw );
}
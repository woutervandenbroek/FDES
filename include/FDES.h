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
#include <malloc.h>
#include <float.h>
#include <math.h>
#include <cufft.h>
#include "paramStructure.h"
#include "rwBinary.h"
#include "optimFunctions.h"
#include "multisliceSimulation.h"
#include "cuda_assert.hpp"
#include "complexMath.h"
#include "coordArithmetic.h"
#include "performanceTimer.h"
#include "projectedPotential.h"
#include "crystalMaker.h"
#include "globalVariables.h"
#include "rwHdf5.h"
#include "rwQsc.h"





#ifdef __linux__
#include <stdbool.h>
#include <getopt.h>
#endif

#ifdef _WIN32
#define required_argument 1
#define no_argument 0
#include "wingetopt.h"
#endif

void showLicense ();



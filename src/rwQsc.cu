/*------------------------------------------------
    Modified from Quantitative TEM/STEM Simulations (QSTEM), author Prof. Christoph Koch.
    url: http://elim.physik.uni-ulm.de/?page_id=834
------------------------------------------------*/

#include "rwQsc.h"

bool readQsc(const char* filename,  params_t** params,int** Z_d, float** xyzCoord_d, float** DWF_d, float** occ_d)
{
  
  if (parOpen(filename) == 0) 
  {
    printf("could not open input file %s!\n",filename);
    exit(0);
  }
  
  
  MULS muls;
  char answer[256];
  FILE *fpTemp;

  char buf[BUF_LEN],*strPtr;
  int i,ix;
  int potDimensions[2];
  double dE_E0,x,y,dx,dy;
  const double pi=3.1415926535897;

  muls.cubex = 0.0;
  muls.cubey = 0.0;
  muls.cubez = 0.0;

  muls.mode = STEM;
  if (readparam("mode:",buf,1)) 
  {
    if (strstr(buf,"TEM")) 
      muls.mode = TEM;
    else
    {
      fprintf (stderr, "\n FDES supports only TEM mode now \n");
      exit(EXIT_FAILURE);
      
    }
      
  }

	muls.printLevel = 2;
	if (readparam("print level:",buf,1)) sscanf(buf,"%d",&(muls.printLevel));
	muls.saveLevel = 0;
	if (readparam("save level:",buf,1)) sscanf(buf,"%d",&(muls.saveLevel));

	
	if ( !readparam( "filename:", buf, 1 ) )
	{
	  	sscanf(buf,"%s",muls.fileBase);
		perror( "Error:" );
		fprintf( stderr,"readFile did not find crystal .cfg file for parsing. \n" );
		exit( 0 );
	}
	sscanf(buf,"%s",muls.fileBase);
	if (muls.fileBase[0] == '"') {
		strPtr = strchr(buf,'"');
		strcpy(muls.fileBase,strPtr+1);
		strPtr = strchr(muls.fileBase,'"');
		*strPtr = '\0';
	}

	if ( readparam( "wavename:", buf, 1) )
	{
		sscanf(buf, "%s", muls.fileWaveIn);
	}
	if ( muls.fileWaveIn[0] == '"' ) 
	{ 	strPtr = strchr( buf, '"' );
		strcpy( muls.fileWaveIn, strPtr + 1 );
		strPtr = strchr( muls.fileWaveIn, '"' );
		*strPtr = '\0';
	}

	if (readparam("NCELLX:",buf,1)) sscanf(buf,"%d",&(muls.nCellX));
	if (readparam("NCELLY:",buf,1)) sscanf(buf,"%d",&(muls.nCellY));

	muls.cellDiv = 1;
	if (readparam("NCELLZ:",buf,1)) {
		sscanf(buf,"%s",answer);
		if ((strPtr = strchr(answer,'/')) != NULL) {
			strPtr[0] = '\0';
			muls.cellDiv = atoi(strPtr+1);
		}
		muls.nCellZ = atoi(answer);
	}
  
	/*************************************************
	* Read the beam tilt parameters
	*/
	muls.btiltx = 0.0;
	muls.btilty = 0.0;
	muls.tiltBack = 1;
	answer[0] = '\0';
	if (readparam("Beam tilt X:",buf,1)) { 
		sscanf(buf,"%g %s",&(muls.btiltx),answer); /* in mrad */
		if (tolower(answer[0]) == 'd')
			muls.btiltx *= pi/180.0;
	}
	answer[0] = '\0';
	if (readparam("Beam tilt Y:",buf,1)) { 
		sscanf(buf,"%g %s",&(muls.btilty),answer); /* in mrad */
		if (tolower(answer[0]) == 'd')
			muls.btilty *= pi/180.0;
	}  
	if (readparam("Tilt back:",buf,1)) { 
		sscanf(buf,"%s",answer);
		muls.tiltBack  = (tolower(answer[0]) == (int)'y');
	}


	/*************************************************
	* Read the crystal tilt parameters
	*/
	muls.ctiltx = 0.0;  /* tilt around X-axis in mrad */
	muls.ctilty = 0.0;  /* tilt around y-axis in mrad */	
	muls.ctiltz = 0.0;  /* tilt around z-axis in mrad */	
	answer[0] = '\0';
	if (readparam("Crystal tilt X:",buf,1)) { 
		sscanf(buf,"%g %s",&(muls.ctiltx),answer); /* in mrad */
		if (tolower(answer[0]) == 'd')
			muls.ctiltx *= pi/180.0;
	}
	answer[0] = '\0';
	if (readparam("Crystal tilt Y:",buf,1)) { 
		sscanf(buf,"%g %s",&(muls.ctilty),answer); /* in mrad */
		if (tolower(answer[0]) == 'd')
			muls.ctilty *= pi/180.0;
	}  
	answer[0] = '\0';
	if (readparam("Crystal tilt Z:",buf,1)) { 
		sscanf(buf,"%g %s",&(muls.ctiltz),answer); /* in mrad */
		if (tolower(answer[0]) == 'd')
			muls.ctiltz *= pi/180.0;
	}
	muls.cubex = 0; muls.cubey = 0; muls.cubez = 0;
	if (readparam("Cube:",buf,1)) { 
		sscanf(buf,"%g %g %g",&(muls.cubex),&(muls.cubey),&(muls.cubez)); /* in A */
	}

	muls.adjustCubeSize = 0;
	if (readparam("Adjust cube size with tilt:",buf,1)) { 
		sscanf(buf,"%s",answer);
		muls.adjustCubeSize  = (tolower(answer[0]) == (int)'y');
	}

	/***************************************************************************
	* temperature related data must be read before reading the atomic positions:
	***************************************************************************/
	if (readparam("tds:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.tds = (tolower(answer[0]) == (int)'y');
	}
	else muls.tds = 0;
	if (readparam("temperature:",buf,1)) sscanf(buf,"%g",&(muls.tds_temp));
	else muls.tds_temp = 300.0;
	muls.Einstein = 1;
	if (readparam("phonon-File:",buf,1)) {
		sscanf(buf,"%s",muls.phononFile);
		muls.Einstein = 0;
	}
 
	/**********************************************************************
	* Read the atomic model positions !!!
	*********************************************************************/
	sprintf(muls.atomPosFile, "%s",muls.fileBase);

	/* remove directory in front of file base: */
	while ((strPtr = strchr(muls.fileBase,'\\')) != NULL) strcpy(muls.fileBase,strPtr+1);

	/* add a '_' to fileBase, if not existent */
	// RAM: ERROR THIS CODE SEEMS TO BREAK IF ONE USES UNDERSCORES IN FILENAME
	// CHANGE TO CHECK FOR UNDERSCORE AT END OF STRING
	if (strrchr(muls.fileBase,'_') != muls.fileBase+strlen(muls.fileBase)-1) {
		if ((strPtr = strchr(muls.fileBase,'.')) != NULL) sprintf(strPtr,"_");
		else strcat(muls.fileBase,"_");
	}

	if (strchr(muls.atomPosFile,'.') == NULL) {
		/*   
		strPtr = strrchr(muls.atomPosFile,'_');
		if (strPtr != NULL)
		*(strPtr) = 0;
		*/
		// take atomPosFile as is, or add an ending to it, if it has none yet
		// RAM: This code can break if there's more than one period in a filename, TO DO: do strcmp's on last four/five characters instead
		if (strrchr(muls.atomPosFile,'.') == NULL) 
		{
			sprintf(buf,"%s.cssr",muls.atomPosFile);
			if ((fpTemp=fopen(buf,"r")) == NULL) 
			{
				sprintf(buf,"%s.cfg",muls.atomPosFile);
				if ((fpTemp=fopen(buf,"r")) == NULL) 
				{
					printf("Could not find input file %s.cssr or %s.cfg\n",
						muls.atomPosFile,muls.atomPosFile);
					exit(0);
				}
				strcat(muls.atomPosFile,".cfg");
				fclose(fpTemp);
			}
			else 
			{
				strcat(muls.atomPosFile,".cssr");
				fclose(fpTemp);
			}
		}
	}
	// We need to initialize a few variables, before reading the atomic 
	// positions for the first time.
	muls.natom = 0;
	muls.atoms = NULL;
	muls.Znums = NULL;
	muls.atomKinds = 0;
	muls.u2 = NULL;
	muls.u2avg = NULL;

	muls.xOffset = 0.0; /* slize z-position offset in cartesian coords */
	if (readparam("xOffset:",buf,1)) sscanf(buf,"%g",&(muls.xOffset));
	muls.yOffset = 0.0; /* slize z-position offset in cartesian coords */
	if (readparam("yOffset:",buf,1)) sscanf(buf,"%g",&(muls.yOffset));
	// printf("Reading Offset: %f, %f\n",muls.xOffset,muls.yOffset);

	// the last parameter is handleVacancies.  If it is set to 1 vacancies 
	// and multiple occupancies will be handled. 
	// _CrtSetDbgFlag  _CRTDBG_CHECK_ALWAYS_DF();
	// printf("memory check: %d, ptr= %d\n",_CrtCheckMemory(),(int)malloc(32*sizeof(char)));

	muls.atoms = readUnitCell(&(muls.natom),muls.atomPosFile,&muls,1);

 


	if (muls.atoms == NULL) {
		printf("Error reading atomic positions!\n");
		exit(0);
	}
	if (muls.natom == 0) {
		printf("No atom within simulation boundaries!\n");
		exit(0);
	}

	/*****************************************************************
	* Done reading atomic positions 
	****************************************************************/

	if (!readparam("nx:",buf,1)) exit(0); sscanf(buf,"%d",&(muls.nx));
	if (readparam("ny:",buf,1)) sscanf(buf,"%d",&(muls.ny));
	else muls.ny = muls.nx;

	muls.resolutionX = 0.0;
	muls.resolutionY = 0.0;
	if (readparam("resolutionX:",buf,1)) sscanf(buf,"%g",&(muls.resolutionX));
	if (readparam("resolutionY:",buf,1)) sscanf(buf,"%g",&(muls.resolutionY));
	if (!readparam("v0:",buf,1)) exit(0); sscanf(buf,"%g",&(muls.v0));

	muls.centerSlices = 0;
	if (readparam("center slices:",buf,1)) {
		// answer[0] =0;
		sscanf(buf,"%s",answer);
		// printf("center: %s (%s)\n",answer,buf);
		muls.centerSlices = (tolower(answer[0]) == (int)'y');
	}
	// just in case the answer was not exactly 1 or 0:
	// muls.centerSlices = (muls.centerSlices) ? 1 : 0;

	muls.sliceThickness = 0.0;
	if (readparam("slice-thickness:",buf,1)) {
		sscanf(buf,"%g",&(muls.sliceThickness));
		if (readparam("slices:",buf,1)) {
			sscanf(buf,"%d",&(muls.slices));
		}
		else {
			if (muls.cubez >0)
				muls.slices = (int)(muls.cubez/(muls.cellDiv*muls.sliceThickness)+0.99);
			else
				muls.slices = (int)(muls.c/(muls.cellDiv*muls.sliceThickness)+0.99);
		}
		muls.slices += muls.centerSlices;
	}
	else {
		muls.slices = 0; 
		if (readparam("slices:",buf,1)) {
			sscanf(buf,"%d",&(muls.slices));
			// muls.slices = (int)(muls.slices*muls.nCellZ/muls.cellDiv);
			if (muls.sliceThickness == 0.0) {
				if ((muls.slices == 1) && (muls.cellDiv == 1)) {
					if (muls.cubez >0)
						muls.sliceThickness = (muls.centerSlices) ? 2.0*muls.cubez/muls.cellDiv : muls.cubez/muls.cellDiv;
					else
						// muls.sliceThickness = (muls.centerSlices) ? 2.0*muls.c/(muls.cellDiv) : muls.c/(muls.cellDiv);
						muls.sliceThickness = (muls.centerSlices) ? 1.0*muls.c/(muls.cellDiv) : muls.c/(muls.cellDiv);
				}
				else {
					if (muls.cubez >0) {
						muls.sliceThickness = muls.cubez/(muls.cellDiv*muls.slices-muls.centerSlices);
					}
					else {
						muls.sliceThickness = muls.c/(muls.cellDiv*muls.slices);
					}
				}
			}
			else {
				muls.cellDiv = (muls.cubez >0) ? (int)ceil(muls.cubez/(muls.slices*muls.sliceThickness)) :
					(int)ceil(muls.c/(muls.slices*muls.sliceThickness));
			if (muls.cellDiv < 1) muls.cellDiv = 1;
			}
		}
	}
	if (muls.slices == 0) {
		if (muls.printLevel > 0) printf("Error: Number of slices = 0\n");
		exit(0);
	}
	/* Find out whether we need to recalculate the potential every time, or not
	*/

	muls.equalDivs = ((!muls.tds)  && (muls.nCellZ % muls.cellDiv == 0) && 
		(fabs(muls.slices*muls.sliceThickness-muls.c/muls.cellDiv) < 1e-5));

	// read the output interval:
	muls.outputInterval = muls.slices;
	if (readparam("slices between outputs:",buf,1)) sscanf(buf,"%d",&(muls.outputInterval));
	if (muls.outputInterval < 1) muls.outputInterval= muls.slices;



	initMuls(&muls);
	muls.czOffset = 0.0; /* slize z-position offset in cartesian coords */
	if (readparam("zOffset:",buf,1)) sscanf(buf,"%g",&(muls.czOffset));



	/***********************************************************************
	* Fit the resolution to the wave function array, if not specified different
	*/
	if (muls.resolutionX == 0.0)
		muls.resolutionX = muls.ax / (double)muls.nx;
	if (muls.resolutionY == 0.0)
		muls.resolutionY = muls.by / (double)muls.ny;



	/************************************************************************
	* Optional parameters:
	* determine whether potential periodic or not, etc.:
	*/
	muls.nonPeriodZ = 1;
	muls.nonPeriod = 1;
	if (readparam("periodicXY:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.nonPeriod = (tolower(answer[0]) != (int)'y');
	}
	if (readparam("periodicZ:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.nonPeriodZ = (tolower(answer[0]) != (int)'y'); /* if 'y' -> nonPeriodZ=0 */
	}
	if ((muls.nonPeriodZ == 0) && (muls.cellDiv > 1)) {
		printf("****************************************************************\n"
			"* Warning: cannot use cell divisions >1 and Z-periodic potential\n"
			"* periodicZ = NO\n"
			"****************************************************************\n");
		muls.nonPeriodZ = 1;
	}

	muls.bandlimittrans = 1;
	if (readparam("bandlimit f_trans:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.bandlimittrans = (tolower(answer[0]) == (int)'y');
	}    
	muls.readPotential = 0;
	if (readparam("read potential:",buf,1)) {
		sscanf(buf," %s",answer);
		muls.readPotential = (tolower(answer[0]) == (int)'y');
	}  
	muls.savePotential = 0;
	if (readparam("save potential:",buf,1)) {
		sscanf(buf," %s",answer);
		muls.savePotential = (tolower(answer[0]) == (int)'y');
	}  
	muls.saveTotalPotential = 0;
	if (readparam("save projected potential:",buf,1)) {
		sscanf(buf," %s",answer);
		muls.saveTotalPotential = (tolower(answer[0]) == (int)'y');
	}  
	muls.plotPotential = 0;
	if (readparam("plot V(r)*r:",buf,1)) {
		sscanf(buf," %s",answer);
		muls.plotPotential = (tolower(answer[0]) == (int)'y');
	}  
	muls.fftpotential = 1;
	if (readparam("one time integration:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.fftpotential = (tolower(answer[0]) == (int)'y');
	}
	muls.potential3D = 1;
	if (readparam("potential3D:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.potential3D = (tolower(answer[0]) == (int)'y');
	}
	muls.avgRuns = 10;
	if (readparam("Runs for averaging:",buf,1))
		sscanf(buf,"%d",&(muls.avgRuns));

	muls.storeSeries = 0;
	if (readparam("Store TDS diffr. patt. series:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.storeSeries = (tolower(answer[0]) == (int)'y');
	}  

	if (!muls.tds) muls.avgRuns = 1;

	muls.scanXStart = muls.ax/2.0;
	muls.scanYStart = muls.by/2.0;
	muls.scanXN = 1;
	muls.scanYN = 1;
	muls.scanXStop = muls.scanXStart;
	muls.scanYStop = muls.scanYStart;


	switch (muls.mode) {
		/////////////////////////////////////////////////////////
		// read the position for doing CBED: 
	case CBED:
		if (readparam("scan_x_start:",buf,1)) sscanf(buf,"%g",&(muls.scanXStart));
		if (readparam("scan_y_start:",buf,1)) sscanf(buf,"%g",&(muls.scanYStart));
		muls.scanXStop = muls.scanXStart;
		muls.scanYStop = muls.scanYStart;
		break;
		/////////////////////////////////////////////////////////
		// read the position for doing NBED: 
	case NBED:
		if (readparam("scan_x_start:", buf, 1)) sscanf(buf, "%g", &(muls.scanXStart));
		if (readparam("scan_y_start:", buf, 1)) sscanf(buf, "%g", &(muls.scanYStart));
		muls.scanXStop = muls.scanXStart;
		muls.scanYStop = muls.scanYStart;
		break;

		/////////////////////////////////////////////////////////
		// Read STEM scanning parameters 

	case STEM:
		/* Read in scan coordinates: */
		if (!readparam("scan_x_start:",buf,1)) exit(0); 
		sscanf(buf,"%g",&(muls.scanXStart));
		if (!readparam("scan_x_stop:",buf,1)) exit(0); 
		sscanf(buf,"%g",&(muls.scanXStop));
		if (!readparam("scan_x_pixels:",buf,1)) exit(0); 
		sscanf(buf,"%d",&(muls.scanXN));
		if (!readparam("scan_y_start:",buf,1)) exit(0); 
		sscanf(buf,"%g",&(muls.scanYStart));
		if (!readparam("scan_y_stop:",buf,1)) exit(0); 
		sscanf(buf,"%g",&(muls.scanYStop));
		if (!readparam("scan_y_pixels:",buf,1)) exit(0); 
		sscanf(buf,"%d",&(muls.scanYN));

		if (muls.scanXN < 1) muls.scanXN = 1;
		if (muls.scanYN < 1) muls.scanYN = 1;
		// if (muls.scanXStart > muls.scanXStop) muls.scanXN = 1;

		muls.displayProgInterval = muls.scanYN*muls.scanYN;
		if (readparam("propagation progress interval:",buf,1)) 
			sscanf(buf,"%d",&(muls.displayProgInterval));
	}
	muls.displayPotCalcInterval = 100000; // RAM: default, but normally read-in by .CFG file in next code fragment
	if ( readparam( "potential progress interval:", buf, 1 ) )
	{
		sscanf( buf, "%d", &(muls.displayPotCalcInterval) );
	}

	/**********************************************************************
	* Read STEM/CBED probe parameters 
	*/
	muls.dE_E = 0.0;
	muls.dI_I = 0.0;
	muls.dV_V = 0.0;
	muls.Cc = 0.0;
	if (readparam("dE/E:",buf,1))
		muls.dE_E = atof(buf);
	if (readparam("dI/I:",buf,1))
		muls.dI_I = atof(buf);
	if (readparam("dV/V:",buf,1))
		muls.dV_V = atof(buf);
	if (readparam("Cc:",buf,1))
		muls.Cc = 1e7*atof(buf);


	/* memorize dE_E0, and fill the array of well defined energy deviations */
	dE_E0 = sqrt(muls.dE_E*muls.dE_E+
		muls.dI_I*muls.dI_I+
		muls.dV_V*muls.dV_V);
	muls.dE_EArray = (double *)malloc((muls.avgRuns+1)*sizeof(double));
	muls.dE_EArray[0] = 0.0;

	/**********************************************************
	* quick little fix to calculate gaussian energy distribution
	* without using statistics (better for only few runs)
	*/
	if (muls.printLevel > 0) printf("avgRuns: %d\n",muls.avgRuns);
	// serious bug in Visual C - dy comes out enormous.
	//dy = sqrt((double)pi)/((double)2.0*(double)(muls.avgRuns));
	// using precalculated sqrt(pi):
	dy = 1.772453850905/((double)2.0*(double)(muls.avgRuns));
	dx = pi/((double)(muls.avgRuns+1)*20);
	for (ix=1,x=0,y=0;ix<muls.avgRuns;x+=dx) {
		y += exp(-x*x)*dx;
		if (y>=ix*dy) {
			muls.dE_EArray[ix++] = x*2*dE_E0/pi;
			if (muls.printLevel > 2) printf("dE[%d]: %g eV\n",ix,muls.dE_EArray[ix-1]*muls.v0*1e3);
			if (ix < muls.avgRuns) {
				muls.dE_EArray[ix] = -muls.dE_EArray[ix-1];
				ix ++;
				if (muls.printLevel > 2) printf("dE[%d]: %g eV\n",ix,muls.dE_EArray[ix-1]*muls.v0*1e3);
			}
		}
	}


	if (!readparam("Cs:",buf,1))  exit(0); 
	sscanf(buf,"%g",&(muls.Cs)); /* in mm */
	muls.Cs *= 1.0e7; /* convert Cs from mm to Angstroem */

	muls.C5 = 0;
	if (readparam("C5:",buf,1)) { 
		sscanf(buf,"%g",&(muls.C5)); /* in mm */
		muls.C5 *= 1.0e7; /* convert C5 from mm to Angstroem */
	}

	/* assume Scherzer defocus as default */
	muls.df0 = -(float)sqrt(1.5*muls.Cs*(wavelength(muls.v0))); /* in A */
	muls.Scherzer = 1;
	if (readparam("defocus:",buf,1)) { 
		sscanf(buf,"%s",answer);
		/* if Scherzer defocus */
		if (tolower(answer[0]) == 's') {
			muls.df0 = -(float)sqrt(1.5*muls.Cs*(wavelength(muls.v0)));
			muls.Scherzer = 1;
		}
		else if (tolower(answer[0]) == 'o') {
			muls.df0 = -(float)sqrt(muls.Cs*(wavelength(muls.v0)));
			muls.Scherzer = 2;
		}
		else {
			sscanf(buf,"%g",&(muls.df0)); /* in nm */
			muls.df0 = 10.0*muls.df0;       /* convert defocus to A */
			muls.Scherzer = (-(float)sqrt(1.5*muls.Cs*(wavelength(muls.v0)))==muls.df0);
		}
	}
	// Astigmatism:
	muls.astigMag = 0;
	if (readparam("astigmatism:",buf,1)) sscanf(buf,"%g",&(muls.astigMag)); 
	// convert to A from nm:
	muls.astigMag = 10.0*muls.astigMag;
	muls.astigAngle = 0;
	if (readparam("astigmatism angle:",buf,1)) sscanf(buf,"%g",&(muls.astigAngle)); 
	// convert astigAngle from deg to rad:
	muls.astigAngle *= pi/180.0;

	////////////////////////////////////////////////////////
	// read in more aberrations:
	muls.a33 = 0;
	muls.a31 = 0;
	muls.a44 = 0;
	muls.a42 = 0;
	muls.a55 = 0;
	muls.a53 = 0;
	muls.a51 = 0;
	muls.a66 = 0;
	muls.a64 = 0;
	muls.a62 = 0;

	muls.phi33 = 0;
	muls.phi31 = 0;
	muls.phi44 = 0;
	muls.phi42 = 0;
	muls.phi55 = 0;
	muls.phi53 = 0;
	muls.phi51 = 0;
	muls.phi66 = 0;
	muls.phi64 = 0;
	muls.phi62 = 0;

	if (readparam("a_33:",buf,1)) {sscanf(buf,"%g",&(muls.a33)); }
	if (readparam("a_31:",buf,1)) {sscanf(buf,"%g",&(muls.a31)); }
	if (readparam("a_44:",buf,1)) {sscanf(buf,"%g",&(muls.a44)); }
	if (readparam("a_42:",buf,1)) {sscanf(buf,"%g",&(muls.a42)); }
	if (readparam("a_55:",buf,1)) {sscanf(buf,"%g",&(muls.a55)); }
	if (readparam("a_53:",buf,1)) {sscanf(buf,"%g",&(muls.a53)); }
	if (readparam("a_51:",buf,1)) {sscanf(buf,"%g",&(muls.a51)); }
	if (readparam("a_66:",buf,1)) {sscanf(buf,"%g",&(muls.a66)); }
	if (readparam("a_64:",buf,1)) {sscanf(buf,"%g",&(muls.a64)); }
	if (readparam("a_62:",buf,1)) {sscanf(buf,"%g",&(muls.a62)); }

	if (readparam("phi_33:",buf,1)) {sscanf(buf,"%g",&(muls.phi33)); }
	if (readparam("phi_31:",buf,1)) {sscanf(buf,"%g",&(muls.phi31)); }
	if (readparam("phi_44:",buf,1)) {sscanf(buf,"%g",&(muls.phi44)); }
	if (readparam("phi_42:",buf,1)) {sscanf(buf,"%g",&(muls.phi42)); }
	if (readparam("phi_55:",buf,1)) {sscanf(buf,"%g",&(muls.phi55)); }
	if (readparam("phi_53:",buf,1)) {sscanf(buf,"%g",&(muls.phi53)); }
	if (readparam("phi_51:",buf,1)) {sscanf(buf,"%g",&(muls.phi51)); }
	if (readparam("phi_66:",buf,1)) {sscanf(buf,"%g",&(muls.phi66)); }
	if (readparam("phi_64:",buf,1)) {sscanf(buf,"%g",&(muls.phi64)); }
	if (readparam("phi_62:",buf,1)) {sscanf(buf,"%g",&(muls.phi62)); }

	muls.phi33 /= (float)RAD2DEG;
	muls.phi31 /= (float)RAD2DEG;
	muls.phi44 /= (float)RAD2DEG;
	muls.phi42 /= (float)RAD2DEG;
	muls.phi55 /= (float)RAD2DEG;
	muls.phi53 /= (float)RAD2DEG;
	muls.phi51 /= (float)RAD2DEG;
	muls.phi66 /= (float)RAD2DEG;
	muls.phi64 /= (float)RAD2DEG;
	muls.phi62 /= (float)RAD2DEG;


	if (!readparam("alpha:",buf,1)) exit(0); 
	sscanf(buf,"%g",&(muls.alpha)); /* in mrad */

	muls.aAIS = 0;  // initialize AIS aperture to 0 A
	if (readparam("AIS aperture:",buf,1)) 
		sscanf(buf,"%g",&(muls.aAIS)); /* in A */

	///// read beam current and dwell time ///////////////////////////////
	muls.beamCurrent = 1;  // pico Ampere
	muls.dwellTime = 1;    // msec
	if (readparam("beam current:",buf,1)) { 
		sscanf(buf,"%g",&(muls.beamCurrent)); /* in pA */
	}
	if (readparam("dwell time:",buf,1)) { 
		sscanf(buf,"%g",&(muls.dwellTime)); /* in msec */
	}
	muls.electronScale = muls.beamCurrent*muls.dwellTime*MILLISEC_PICOAMP;
	//////////////////////////////////////////////////////////////////////

	muls.sourceRadius = 0;
	if (readparam("Source Size (diameter):",buf,1)) 
		muls.sourceRadius = atof(buf)/2.0;

	if (readparam("smooth:",buf,1)) sscanf(buf,"%s",answer);
	muls.ismoth = (tolower(answer[0]) == (int)'y');
	muls.gaussScale = 0.05f;
	muls.gaussFlag = 0;
	if (readparam("gaussian:",buf,1)) {
		sscanf(buf,"%s %g",answer,&(muls.gaussScale));
		muls.gaussFlag = (tolower(answer[0]) == (int)'y');
	}

	/**********************************************************************
	* Parameters for image display and directories, etc.
	*/
	muls.imageGamma = 1.0;
	if (readparam("Display Gamma:",buf,1)) {
		muls.imageGamma = atof(buf);
	}
	muls.showProbe = 0;
	if (readparam("show Probe:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.showProbe = (tolower(answer[0]) == (int)'y');
	}

	// Setup folder for saved data
	sprintf(muls.folder,"data");
	if (readparam("Folder:",buf,1)) 
		sscanf(buf," %s",muls.folder);

	// search for second '"', in case the folder name is in quotation marks:
// 	printf( "stem3.readFile: folder is %s \n", muls.folder );
	if ( muls.folder[0] == '"' ) {
		strPtr = strchr( buf, '"' );
		strcpy( muls.folder, strPtr + 1 );
		strPtr = strchr( muls.folder, '"' );
		*strPtr = '\0';
	}

	if ( muls.folder[strlen( muls.folder ) - 1] == '/' )
	{
		muls.folder[strlen( muls.folder ) - 1] = '\0';
	}

	// Web update
	muls.webUpdate = 0;
	if (readparam("update Web:",buf,1)) {
		sscanf(buf,"%s",answer);
		muls.webUpdate = (tolower(answer[0]) == (int)'y');
	}



	/*  readBeams(parFp); */  
	/************************************************************************/  
	/* read the different detector configurations                           */
	resetParamFile();
	muls.detectorNum = 0;

	if (muls.mode == STEM) 
	{
		int tCount = (int)(ceil((double)((muls.slices * muls.cellDiv) / muls.outputInterval)));

		/* first determine number of detectors */
		while (readparam("detector:",buf,0)) muls.detectorNum++;  
		/* now read in the list of detectors: */
		resetParamFile();

		// loop over thickness planes where we're going to record intermediates
		// TODO: is this too costly in terms of memory?  It simplifies the parallelization to
		//       save each of the thicknesses in memory, then save to disk afterwards.
		for (int islice=0; islice<=tCount; islice++)
		{
			std::vector<DetectorPtr> detectors;
			resetParamFile();
			while (readparam("detector:",buf,0)) {
				DetectorPtr det = DetectorPtr(new Detector(muls.scanXN, muls.scanYN, 
					(muls.scanXStop-muls.scanXStart)/(float)muls.scanXN,
					(muls.scanYStop-muls.scanYStart)/(float)muls.scanYN));
				
				sscanf(buf,"%g %g %s %g %g",&(det->rInside),
					&(det->rOutside), det->name, &(det->shiftX),&(det->shiftY));  

				/* determine v0 specific k^2 values corresponding to the angles */
				det->k2Inside = 
					(float)(sin(det->rInside*0.001)/(wavelength(muls.v0)));
				det->k2Outside = 
					(float)(sin(det->rOutside*0.001)/(wavelength(muls.v0)));
				// printf("Detector %d: %f .. %f, lambda = %f (%f)\n",i,muls.detectors[i]->k2Inside,muls.detectors[i]->k2Outside,wavelength(muls.v0),muls.v0);
				/* calculate the squares of the ks */
				det->k2Inside *= det->k2Inside;
				det->k2Outside *= det->k2Outside;
				detectors.push_back(det);
			}
			muls.detectors.push_back(detectors);
		}
	}
	/************************************************************************/   

	// in case this file has been written by the tomography function, read the current tilt:
	if (readparam("tomo tilt:",buf,1)) { 
		sscanf(buf,"%lf %s",&(muls.tomoTilt),answer); /* in mrad */
		if (tolower(answer[0]) == 'd')
			muls.tomoTilt *= 1000*pi/180.0;
	}
	/************************************************************************
	* Tomography Parameters:
	***********************************************************************/
	if (muls.mode == TOMO) {     
		if (readparam("tomo start:",buf,1)) { 
			sscanf(buf,"%lf %s",&(muls.tomoStart),answer); /* in mrad */
			if (tolower(answer[0]) == 'd')
				muls.tomoStart *= 1000*pi/180.0;
		}
		if (readparam("tomo step:",buf,1)) {
			sscanf(buf,"%lf %s",&(muls.tomoStep),answer); /* in mrad */
			if (tolower(answer[0]) == 'd')
				muls.tomoStep *= 1000*pi/180.0;
		}

		if (readparam("tomo count:",buf,1))  
			muls.tomoCount = atoi(buf); 
		if (readparam("zoom factor:",buf,1))  
			sscanf(buf,"%lf",&(muls.zoomFactor));
		if ((muls.tomoStep == 0) && (muls.tomoStep > 1))
			muls.tomoStep = -2.0*muls.tomoStart/(double)(muls.tomoCount - 1);
	}
	/***********************************************************************/


	/*******************************************************************
	* Read in parameters related to the calculation of the projected
	* Potential
	*******************************************************************/
	muls.atomRadius = 5.0;
	if (readparam("atom radius:",buf,1))  
		sscanf(buf,"%g",&(muls.atomRadius)); /* in A */
	// why ??????  so that number of subdivisions per slice >= number of fitted points!!!
	/*  
	if (muls.atomRadius < muls.sliceThickness)
	muls.atomRadius = muls.sliceThickness;
	*/
	muls.scatFactor = DOYLE_TURNER;
	if (readparam("Structure Factors:",buf,1)) {
		sscanf(buf," %s",answer);
		switch (tolower(answer[0])) {
	case 'w':
		if (tolower(answer[1])=='k') muls.scatFactor = WEICK_KOHL;
		break;
	case 'd':  // DOYLE_TURNER
		muls.scatFactor = DOYLE_TURNER;
		break;
	case 'c':  // CUSTOM - specify k-lookup table and values for all atoms used
		muls.scatFactor = CUSTOM;
		// we already have the kinds of atoms stored in 
		// int *muls.Znums and int muls.atomKinds
		readSFactLUT(&muls);
		break;
	default:
		muls.scatFactor = DOYLE_TURNER;
		}
	}




	/***************************************************************
	* We now need to determine the size of the potential array, 
	* and the offset from its edges in A.  We only need to calculate
	* as much potential as we'll be illuminating later with the 
	* electron beam.
	**************************************************************/
	muls.potOffsetX = 0;
	muls.potOffsetY = 0;

	if ((muls.mode == STEM) || (muls.mode == CBED)) {
		/* we are assuming that there is enough atomic position data: */
		muls.potOffsetX = muls.scanXStart - 0.5*muls.nx*muls.resolutionX;
		muls.potOffsetY = muls.scanYStart - 0.5*muls.ny*muls.resolutionY;
		/* adjust scanStop so that it coincides with a full pixel: */
		muls.potNx = (int)((muls.scanXStop-muls.scanXStart)/muls.resolutionX);
		muls.potNy = (int)((muls.scanYStop-muls.scanYStart)/muls.resolutionY);
		muls.scanXStop = muls.scanXStart+muls.resolutionX*muls.potNx;
		muls.scanYStop = muls.scanYStart+muls.resolutionY*muls.potNy;
		muls.potNx+=muls.nx;
		muls.potNy+=muls.ny;
		muls.potSizeX = muls.potNx*muls.resolutionX;
		muls.potSizeY = muls.potNy*muls.resolutionY;
	}
	else {
		muls.potNx = muls.nx;
		muls.potNy = muls.ny;
		muls.potSizeX = muls.potNx*muls.resolutionX;
		muls.potSizeY = muls.potNy*muls.resolutionY;
		muls.potOffsetX = muls.scanXStart - 0.5*muls.potSizeX;
		muls.potOffsetY = muls.scanYStart - 0.5*muls.potSizeY;
	}  
	/**************************************************************
	* Check to see if the given scan parameters really fit in cell 
	* dimensions:
	*************************************************************/
	if ((muls.scanXN <=0) ||(muls.scanYN <=0)) {
		printf("The number of scan pixels must be >=1\n");
		exit(0);
	}
	if ((muls.scanXStart<0) || (muls.scanYStart<0) ||
		(muls.scanXStop<0) || (muls.scanYStop<0) ||
		(muls.scanXStart>muls.ax) || (muls.scanYStart>muls.by) ||
		(muls.scanXStop>muls.ax) || (muls.scanYStop>muls.by)) {
			printf("Scanning window is outside model dimensions (%g,%g .. %g,%g) [ax = %g, by = %g]!\n",muls.scanXStart,muls.scanYStart,muls.scanXStop,muls.scanYStop,muls.ax,muls.by);
			exit(0);
	}
	/*************************************************************
	* read in the beams we want to plot in the pendeloesung plot
	* Only possible if not in STEM or CBED mode 
	*************************************************************/
	muls.lbeams = 0;   /* flag for beam output */
	muls.nbout = 0;    /* number of beams */
	resetParamFile();
	if ((muls.mode != STEM) && (muls.mode != CBED)) {
		if (readparam("Pendelloesung plot:",buf,1)) {
			sscanf(buf,"%s",answer);
			muls.lbeams = (tolower(answer[0]) == (int)'y');
		}
		if (muls.lbeams) {
			while (readparam("beam:",buf,0)) muls.nbout++;  
			printf("will record %d beams\n",muls.nbout);
			muls.hbeam = (int*)malloc(muls.nbout*sizeof(int));
			muls.kbeam = (int*)malloc(muls.nbout*sizeof(int));
			/* now read in the list of detectors: */
			resetParamFile();
			for (i=0;i<muls.nbout;i++) {
				if (!readparam("beam:",buf,0)) break;
				muls.hbeam[i] = 0;
				muls.kbeam[i] = 0;
				sscanf(buf,"%d %d",muls.hbeam+i,muls.kbeam+i);
				muls.hbeam[i] *= muls.nCellX;
				muls.kbeam[i] *= muls.nCellY;

				muls.hbeam[i] = (muls.hbeam[i]+muls.nx) % muls.nx;
				muls.kbeam[i] = (muls.kbeam[i]+muls.ny) % muls.ny;
				printf("beam %d [%d %d]\n",i,muls.hbeam[i],muls.kbeam[i]); 			}
		}
	}

	/* TODO: possible breakage here - MCS 2013/04 - made muls.cfgFile be allocated on the struct
	       at runtim - thus this null check doesn't make sense anymore.  Change cfgFile set
	   Old comment:
		if cfgFile != NULL, the program will later write a the atomic config to this file */
	//muls.cfgFile = NULL;
	// RAM Apr14: Fix this broken code by adding an else statement to generate a default cfg-file extension for TDS/tilted specimens
	if ( readparam( "CFG-file:", buf, 1 ) )
	{
// 		printf( "Debug, MSC breakge for filename: %s", muls.cfgFile );
		sscanf( buf, "%s", muls.cfgFile );
	}
	else
	{
		// RAM: Build a default filename with added _t to indicated tilted or tds crystal.
		strcpy( muls.cfgFile, muls.fileBase );
		strcat( muls.cfgFile, "t" );
// 		printf( "DEBUG: tilt/tds default filename made = %s \n", muls.cfgFile );
	}

	/* allocate memory for wave function */

	potDimensions[0] = muls.potNx;
	potDimensions[1] = muls.potNy;
	
	int fftMeasureFlag =FFTW_ESTIMATE;
#if FLOAT_PRECISION == 1
	muls.trans = complex3Df(muls.slices,muls.potNx,muls.potNy,"trans");
	// printf("allocated trans %d %d %d\n",muls.slices,muls.potNx,muls.potNy);
	muls.fftPlanPotForw = fftwf_plan_many_dft(2,potDimensions, muls.slices,muls.trans[0][0], NULL,
		1, muls.potNx*muls.potNy,muls.trans[0][0], NULL,
		1, muls.potNx*muls.potNy, FFTW_FORWARD, fftMeasureFlag);
	muls.fftPlanPotInv = fftwf_plan_many_dft(2,potDimensions, muls.slices,muls.trans[0][0], NULL,
		1, muls.potNx*muls.potNy,muls.trans[0][0], NULL,
		1, muls.potNx*muls.potNy, FFTW_BACKWARD, fftMeasureFlag);
#else

	muls.trans = complex3D(muls.slices,muls.potNx,muls.potNy,"trans");
	muls.fftPlanPotForw = fftw_plan_many_dft(2,potDimensions, muls.slices,muls.trans[0][0], NULL,
		1, muls.potNx*muls.potNy,muls.trans[0][0], NULL,
		1, muls.potNx*muls.potNy, FFTW_FORWARD, fftMeasureFlag);
	muls.fftPlanPotInv = fftw_plan_many_dft(2,potDimensions, muls.slices,muls.trans[0][0], NULL,
		1, muls.potNx*muls.potNy,muls.trans[0][0], NULL,
		1, muls.potNx*muls.potNy, FFTW_BACKWARD, fftMeasureFlag);
#endif

	////////////////////////////////////
	if (muls.printLevel >= 4) 
		printf("Memory for transmission function (%d x %d x %d) allocated and plans initiated\n",muls.slices,muls.potNx,muls.potNy);
	
	
	
////////////////////////////////////////////////
 
	int n3;
	n3=1;

	allocParams ( params, n3);
	defaultParams( * params,n3);
	(**params).IM.n3 = n3;
	(**params).IM.n1 = muls.nx;
	(**params).IM.n2 = muls.ny;
	(**params).IM.dn1 =round(float(muls.nx/2));
	(**params).IM.dn2 =round(float(muls.ny/2));
	(**params).IM.m1 = (**params).IM.n1 + 2* (**params).IM.dn1 ;
	(**params).IM.m2 = (**params).IM.n2 + 2* (**params).IM.dn2 ;
	(**params).IM.m3 = muls.slices;
	(**params).IM.d1 = muls.resolutionX * 1e-10 ; // A-> m
	(**params).IM.d2 = muls.resolutionY * 1e-10 ;
	(**params).IM.d3 = muls.sliceThickness* 1e-10; 
	(**params).IM.subSlTh = muls.sliceThickness* 1e-10/10; 
	(**params).IM.specimen_tilt_offset_x = muls.ctiltx;
	(**params).IM.specimen_tilt_offset_y = muls.ctilty;
	(**params).IM.specimen_tilt_offset_z = muls.ctiltz;
	
	(**params).IM.tiltbeam[0] = muls.btiltx;
	(**params).IM.tiltbeam[1] = muls.btilty;
	
	(**params).EM.E0 = muls.v0 * 1e3; // kv -> v
	(**params).EM.illangle = muls.alpha / 1e3; //mrad -> rad
	
	
	(**params).EM.aberration.A1_0= muls.astigMag * 1e-9;
	(**params).EM.aberration.A1_1= muls.astigAngle * 1e-9;
	(**params).EM.aberration.C1_0= muls.df0* 1e-10; //A -> m
	(**params).EM.aberration.C3_0= muls.Cs * 1e-10; // Angstroem -> m
	(**params).EM.aberration.C5_0= muls.C5 * 1e-3;	
	

	
	strcpy((**params).SAMPLE.material,muls.atomPosFile);
	char *tempChar;
	
	tempChar =strstr ((**params).SAMPLE.material, ".cfg");
	if (tempChar!=NULL)
	{
	  strncpy (tempChar,"",3); 
	}
	else
	{
	  exit(EXIT_FAILURE);
	}
	
	strcpy((**params).SAMPLE.sample_name,muls.atomPosFile);
	
	tempChar =strstr ((**params).SAMPLE.sample_name, ".cfg");
	if (tempChar!=NULL)
	{
	  char cellnum[128];
	  sprintf(cellnum,"_CELL_%02d_%02d_%02d", muls.nCellX, muls.nCellY,muls.nCellZ);
	  strncpy (tempChar,cellnum,strlen(cellnum)); 
	}
	
	if (readparam("cal_mode:",buf,1)) sscanf(buf,"%d",&((**params).IM.mode));
	
	if (readparam("focus_spread:",buf,1)) sscanf(buf,"%g",&((**params).EM.defocspread));
	
	if (readparam("objective_aperture:",buf,1)) sscanf(buf,"%g",&((**params).EM.ObjAp));
	
	if (readparam("pixel_dose:",buf,1)) sscanf(buf,"%g",&((**params).IM.pD));
	
	if (readparam("absorptive_potential_factor:",buf,1)) sscanf(buf,"%g",&((**params).SAMPLE.imPot));
 	
	if (readparam("mtf_a:",buf,1)) sscanf(buf,"%g",&((**params).EM.mtfa));
		
	if (readparam("mtf_b:",buf,1)) sscanf(buf,"%g",&((**params).EM.mtfb));
	
	if (readparam("mtf_c:",buf,1)) sscanf(buf,"%g",&((**params).EM.mtfc));

	if (readparam("frozen_phonons:",buf,1)) sscanf(buf,"%d",&((**params).IM.frPh));
	
	if(!atomsFromExternal)
	{
	int nAt=muls.natom;
	(**params).SAMPLE.nAt = muls.natom;
	
	int *Z_h;
	float *xyzCoord_h, *DWF_h, *occ_h;
	Z_h = ( int* ) malloc ( nAt * sizeof ( int ) );   
	xyzCoord_h = ( float* ) malloc ( nAt * 3 * sizeof ( float ) );
	DWF_h = ( float* ) malloc ( nAt * sizeof ( float ) );
	occ_h = ( float* ) malloc ( nAt * sizeof ( float ) );
  
	cuda_assert ( cudaMalloc ( ( void** ) Z_d,        nAt* sizeof ( int ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) xyzCoord_d, nAt * 3 * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) DWF_d,      nAt * sizeof ( float ) ) );
	cuda_assert ( cudaMalloc ( ( void** ) occ_d,      nAt * sizeof ( float ) ) );

	for (int i=0; i< nAt; i++)
	{
	  Z_h[i]   = muls.atoms[i].Znum;
	  DWF_h[i] = muls.atoms[i].dw*1e-20;
	  occ_h[i] = muls.atoms[i].occ;
	  xyzCoord_h[3*i+0] =muls.atoms[i].x * 1e-10;// from A->m
	  xyzCoord_h[3*i+1] =muls.atoms[i].y * 1e-10;
	  xyzCoord_h[3*i+2] =muls.atoms[i].z * 1e-10;
	}
	
	float minX=1, minY=1,minZ=1;
	float maxX=0, maxY=0,maxZ=0;
	
	for (int i=0; i< nAt; i++)
	{
	  if (xyzCoord_h[3*i+0]>maxX)
	  {
	    maxX = xyzCoord_h[3*i+0];
	  }
	  
	  if(xyzCoord_h[3*i+0] < minX)
	  {
	    minX = xyzCoord_h[3*i+0];
	  }
	  
	  if (xyzCoord_h[3*i+1]>maxY)
	  {
	    maxY = xyzCoord_h[3*i+1];
	  }
	  
	  if(xyzCoord_h[3*i+1] < minY)
	  {
	    minY = xyzCoord_h[3*i+1];
	  }
	  
	  if (xyzCoord_h[3*i+2]>maxZ)
	  {
	    maxZ = xyzCoord_h[3*i+2];
	  }
	  
	  if(xyzCoord_h[3*i+2] < minZ)
	  {
	    minZ = xyzCoord_h[3*i+2];
	  }
  
	}
	
	for (int i=0; i< nAt; i++)
	{
	 xyzCoord_h[3*i+0] = xyzCoord_h[3*i+0] - (maxX-minX)/2 ;
	 xyzCoord_h[3*i+1] = xyzCoord_h[3*i+1] - (maxY-minY)/2 ;
	 xyzCoord_h[3*i+2] = xyzCoord_h[3*i+2] - (maxZ-minZ)/2 ;
	}
	
/////////////////////////////////////////////////////////////////////
	cuda_assert ( cudaMemcpy ( *Z_d,       Z_h ,       nAt*sizeof ( int ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( *xyzCoord_d,xyzCoord_h, nAt* 3 * sizeof ( float ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( *DWF_d,     DWF_h,      nAt *sizeof ( float ), cudaMemcpyHostToDevice ) );
	cuda_assert ( cudaMemcpy ( *occ_d,     occ_h,      nAt* sizeof ( float ), cudaMemcpyHostToDevice ) );
	}
/////////////////////////////////////////////////////////////////////   
	consitentParams ( *params );
	setGridAndBlockSize ( *params );
	setCufftPlan ( *params );
	setCublasHandle ( *params );

	writeConfig("ParamsUsedQsc.txt",*params, Z_d,  xyzCoord_d, DWF_d, occ_d );
  
  
  return true;
}

void initMuls( MULS *muls) {
	int sCount,i,slices;

	slices = muls->slices;

	/* general setup: */
	muls->lpartl = 0;

	muls->atomRadius = 5.0;  /* radius in A for making the potential boxes */

	for (sCount =0;sCount<slices;sCount++)
		muls->cin2[sCount] = 'a'+sCount;
	for (sCount = slices;sCount < NCINMAX;sCount++)
		muls->cin2[sCount] = 0;
	muls->nlayer = slices;
	muls->saveFlag = 0;

	muls->sigmaf = 0;
	muls->dfdelt = 0;
	muls->acmax = 0;
	muls->acmin = 0;
	muls->aobj = 0;
	muls->Cs = 0;
	muls->aAIS = 0;
	// muls->areaAIS = 1.0;

	// Tomography parameters:
	muls->tomoTilt = 0;
	muls->tomoStart = 0;
	muls->tomoStep = 0;
	muls->tomoCount = 0;  // indicate: NO Tomography simulation.

	/* make multislice read the inout files and assign transr and transi: */
	muls->trans = NULL;
	muls->cz = NULL;  // (real *)malloc(muls->slices*sizeof(real));

	muls->onlyFresnel = 0;
	muls->showPhaseplate = 0;
	muls->czOffset = 0;  /* defines the offset for the first slice in 
						fractional coordinates        */
	muls->normHolog = 0;
	muls->gaussianProp = 0;


	muls->sparam = (float *)malloc(NPARAM*sizeof(float));
	for (i=0;i<NPARAM;i++)
		muls->sparam[i] = 0.0;
	muls->kx = NULL;
	muls->kx2= NULL;
	muls->ky = NULL;
	muls->ky2= NULL;

	/****************************************************/
	/* copied from slicecell.c                          */
	muls->pendelloesung = NULL;
}


/***********************************************************************
* readSFactLUT() reads the scattering factor lookup table from the 
* input file
**********************************************************************/
void readSFactLUT( MULS *muls) {
	int Nk,i,j;
	double **sfTable=NULL;
	double *kArray = NULL;
	char buf[256], elem[8];

	if (readparam("Nk:",buf,1))
		Nk = atoi(buf);
	else {
		printf("Could not find number of k-points for custom scattering factors (Nk)\n");
		exit(0);
	}

	// allocate memory for sfTable and kArray:
	sfTable = double2D(muls->atomKinds,Nk+1,"sfTable");
	kArray  = double1D(Nk+1,"kArray");

	// read the k-values:
	readArray("k:",kArray,Nk);
	kArray[Nk] = 2.0*kArray[Nk-1];

	for (j=0;j<muls->atomKinds;j++) {
		elem[3] = '\0';
		elem[2] = ':';
		elem[0] = elTable[2*muls->Znums[j]-2];
		elem[1] = elTable[2*muls->Znums[j]-1];
		if (elem[1] == ' ') {
			elem[1] = ':';
			elem[2] = '\0';
		}
		// printf("%s\n",elem);
		readArray(elem,sfTable[j],Nk);
		sfTable[j][Nk] = 0.0;
	}

	if (0) {
		printf("k: ");
		for (i=0;i<=Nk;i++) printf("%.3f ",kArray[i]);
		for (j=0;j<muls->atomKinds;j++) {
			printf("\n%2d: ",muls->Znums[j]);
			for (i=0;i<=Nk;i++) printf("%.3f ",sfTable[j][i]);
		}
		printf("\n");
	}
	muls->sfTable = sfTable;
	muls->sfkArray = kArray;
	muls->sfNk = Nk+1;
}

void readArray(const char *title,double *array,int N) {
	int i;
	char buf[512],*str;

	if (!readparam(title,buf,1)) printf("%s array not found - exit\n",title), exit(0);
	i=0;
	str = buf;
	if (strchr(" \t\n",*str) != NULL)
	  str = strnext(str," \t"); 
	while (i<N) {
		array[i++] = atof(str);
		str = strnext(str," \t\n");
		while (str == NULL) {
			if (!readNextLine(buf,511)) 
				printf("Incomplete reading of %s array - exit\n",title), exit(0);
			str = buf;
			if (strchr(" \t\n",*str) != NULL) str = strnext(str," \t\n");
		}  
	}
}


double wavelength( double kev )
{
  double w;
  const double emass=510.99906; /* electron rest mass in keV */
  const double hc=12.3984244; /* Planck's const x speed of light*/
  
  /* electron wavelength in Angstroms */
  w = hc/sqrt( kev * ( 2*emass + kev ) );
  
  return( w );
  
}  /* end wavelength() */




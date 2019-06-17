
      PROGRAM TEMAIN
      IMPLICIT NONE

      DOUBLE PRECISION RANDOMSEED
C
C   Maximum number of events in configuration file
      integer MAXEVENTS
      parameter (MAXEVENTS=1000)
      integer EVENTS(MAXEVENTS,3)   ! Step fault on=1/off=0
C  
C  Instructions for running the program
C  ====================================
C
C  1) Go to line 220, change NPTS to the number of data points to simulate. For each
C     minute of operation, 60 points are generated.
      INTEGER NPTS
C      PARAMETER (NPTS = 172800)       ! ORIGINAL
C      PARAMETER (NPTS = 385600)
C
C  2) Go to line 226, change SSPTS to the number of data points to simulate in steady
C     state operation before implementing the disturbance.
      INTEGER SSPTS
C      PARAMETER(SSPTS = 3600 * 8)       ! = 28800   ORIGINAL
C      PARAMETER(SSPTS = 172800)
C
C  3) Go to line 367, implement any of the 21 programmed disturbances. For example, to
C     implement disturbance 2, type IDV(2)=1 .
      INTEGER IDV(20)   ! Multiple fault vector
C                  1 2 3 4 5 6 7 8 9 1011121314151617181920
C      DATA IDV  /0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0/     ! ORIGINAL
C      DATA IDV  /0,1,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0,0,0/
C
C  4) The program will generate 15 output files and all data are recorded every
C     180 seconds, see Table 1 for details.  The default path is the home directory.
C     To change the file name and path, modify lines 346-360 accordingly.  
C     To overwrite the files that already existed, change STATUS='new' to 
C     STATUS='old' from lines 346-360.
C
C
C            
C=============================================================================
C
      INTEGER NUMTS, TIMESTAMP(MAXEVENTS)

      integer argc
      character*80 argv
      integer IARGC

      CHARACTER *80 CONFIGFNAME

      DATA CONFIGFNAME / '../cfg/config.csv' /
      argc = IARGC() ! Number of command line arguments
      if( argc .eq. 0 ) then
          write(*,*) 'Loading default config: ', CONFIGFNAME
      else
          call getarg(1,argv)
          CONFIGFNAME = argv
          write(*,*) 'Loading config:', CONFIGFNAME
      endif
C
C  Read parameters from configuration file
      CALL READCONFIG(CONFIGFNAME,EVENTS,MAXEVENTS,
     +                TIMESTAMP,NUMTS,NPTS,SSPTS,IDV,RANDOMSEED)
C
      CALL TESUB(RANDOMSEED,MAXEVENTS,EVENTS,NPTS,SSPTS,IDV)
C
C

      END

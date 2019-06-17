#include "fintrf.h"      
C======================================================================
C
C	THE FOLLOWING '#if 0' HEADER MUST BE HERE
C
#if 0
C     
C     tesub.F
C     .F file needs to be preprocessed to generate .for equivalent
C     
#endif
C     
C======================================================================
C     Gateway routine
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)

C     Declarations
      implicit none

C     mexFunction arguments:
      mwPointer plhs(*), prhs(*)
      integer nlhs, nrhs

C     Function declarations:
      mwPointer mxGetM, mxGetN ! Get number of rows, columns in array 
      mwPointer m, n
      integer mxIsDouble, mxIsInt8, mxIsUint8, mxIsInt16, mxIsUint16,
     +                    mxIsInt32, mxIsUint32, mxIsInt64, mxIsUint64
      integer argnum
      mwPointer arg
      mwPointer mxGetPr
      mwPointer mxCreateDoubleMatrix, mxCreateDoubleScalar


      real*8, allocatable :: X(:,:), Y(:,:) ! 2-D dynamically allocatable array
      real*8 dblbuf, RANDOMSEED
      integer NPTS, numevents, colevent, TSPERSAMPLE, numclass,
     +        numsamples, numfeat, i, j
      PARAMETER( numfeat=53, TSPERSAMPLE=180, colevent=3, numclass=20 )
C      INTEGER IDV(20)   ! Multiple fault vector
      integer, allocatable :: EVENTS(:,:)
      real*8, allocatable :: eventbuf(:,:)
      character*80 str
      character*46 usage
     +          / 'Usage: [X,Y] = tesub(RANDOMSEED, NPTS, EVENTS)' /
      
C https://www.mathworks.com/help/matlab/matlab_prog/capture-information-about-errors.html#bq9tdlq-1
C      call mexErrMsgIdAndTxt ('Matlab:DEBUG','Test: BYE...')

      if( .true. ) then
      endif
C-----------------------------------------------------------------------
C     Check for proper number of arguments. 

      if (nrhs .ne. 3) then
         call mexErrMsgIdAndTxt ('MATLAB:tesub:nInput',usage)
      elseif (nlhs .ne. 2) then
         call mexErrMsgIdAndTxt ('MATLAB:tesub:nOutput',usage)
      endif

C Check and read first argument = random seed
      argnum = 1
      arg = prhs(argnum)
      m = mxGetM(arg)
      n = mxGetN(arg)
C      write(str,'(1x,3(A,I3),A)') 'argnum=',argnum,' m=',m,' n=',n,'\n'
C      call mexPrintf(str)

      if( m.ne.1 .or. n.ne.1 .or. mxIsDouble(arg).ne.1 ) then
         call mexErrMsgIdAndTxt ('MATLAB:tesub:argtype',
     +               'First argument must be a single double number.')
      endif
      call mxCopyPtrToReal8(mxGetPr(arg), RANDOMSEED, 1)
      write(str,'(1x,A22,E20.10,A)') 'RANDOMSEED=',RANDOMSEED,'\n'
      call mexPrintf(str)
      

C Check and read second argument = number of sampling points within TE simulator
      argnum = 2
      arg = prhs(argnum)
      m = mxGetM(arg)
      n = mxGetN(arg)
C      write(str,'(1x,3(A,I3),A)') 'argnum=',argnum,' m=',m,' n=',n,'\n'
C      call mexPrintf(str)

      if( m.ne.1 .or. n.ne.1 .or. mxIsDouble(arg).ne.1 ) then
         call mexErrMsgIdAndTxt ('MATLAB:tesub:argtype',
     +               'Second argument must be a single integer number.')
      endif
      call mxCopyPtrToReal8(mxGetPr(arg), dblbuf, 1)
C      write(str,'(1x,A,E20.10,A)') 'dblbuf=',dblbuf,'\n'
C      call mexPrintf(str)
      call mxCopyPtrToReal8(mxGetPr(arg), dblbuf, 1)
      NPTS = int(dblbuf)
      write(str,'(1x,A22,I20,A)') 'NPTS=',NPTS,'\n'
      call mexPrintf(str)

C Check and read third argument = Eventlist = (Time, Fault, On/Off)
      argnum = 3
      arg = prhs(argnum)
      m = mxGetM(arg)
      n = mxGetN(arg)
C      write(str,'(1x,3(A,I3),A)') 'argnum=',argnum,' m=',m,' n=',n,'\n'
C      call mexPrintf(str)
      if( n.ne.colevent ) then
         call mexErrMsgIdAndTxt ('MATLAB:tesub:argtype',
     +               'Third argument must have three columns.')
      endif

      numevents = m
      write(str,'(1x,A22,I20,A)') 'NUMBER OF EVENTS=',numevents,'\n'
      call mexPrintf(str)

C Read events from eventmatrix and put it into variable EVENTS
      allocate( eventbuf(numevents,colevent) )
      allocate( EVENTS(numevents,colevent) )
      call mxCopyPtrToReal8(mxGetPr(arg),eventbuf,numevents*colevent)
      do i=1,numevents
        do j=1,colevent
          EVENTS(i,j) = int(eventbuf(i,j))
        enddo
        write(str,'(1x,A,I5,A,3(I16),A)') 'event[',i,']=',
     +              (EVENTS(i,j),j=1,colevent),'\n'
        call mexPrintf(str)
      enddo
      deallocate( eventbuf )

      numsamples = NPTS / TSPERSAMPLE
      allocate( X(numsamples,numfeat) )
      allocate( Y(numsamples,numclass) )

C     Call the computational subroutine.

      call tesub(RANDOMSEED,NPTS,TSPERSAMPLE,
     +     numsamples,numfeat,numclass,X,Y,numevents,colevent,EVENTS)

      plhs(1) = mxCreateDoubleMatrix(numsamples,numfeat, 0)
      call mxCopyReal8ToPtr(X,mxGetPr(plhs(1)),numsamples*numfeat)

      plhs(2) = mxCreateDoubleMatrix(numsamples,numclass, 0)
      call mxCopyReal8ToPtr(Y,mxGetPr(plhs(2)),numsamples*numclass)

      deallocate( Y )
      deallocate( X )
      deallocate( EVENTS )
      return
      end


      subroutine teinfo(str)
        character str*80
        character teprompt*5
C        return ! be quite
        write(teprompt,'(1x,A4)') 'TE> '
        call mexPrintf(teprompt)
        call mexPrintf(str)
      end

C               Tennessee Eastman Process Control Test Problem
C
C                  Original codes written by
C
C                    James J. Downs and Ernest F. Vogel
C
C                  Process and Control Systems Engineering
C                        Tennessee Eastman Company
C                              P.O. Box 511
C                          Kingsport, Tennessee 37662
C
C--------------------------------------------------------------------
C
C  New version is a closed-loop plant-wide control scheme for 
C  the Tennessee Eastman Process Control Test Problem
C                        
C  The modifications are by:
C
C            Evan L. Russell, Leo H. Chiang and Richard D. Braatz
C
C                 Large Scale Systems Research Laboratory
C                      Department of Chemical Engineering
C                University of Illinois at Urbana-Champaign
C                       600 South Mathews Avenue, Box C-3
C                        Urbana, Illinois 61801
C                       http://brahms.scs.uiuc.edu
C
C The modified text is Copyright 1998-2002 by The Board of Trustees 
C of the University of Illinois.  All rights reserved.
C 
C Permission hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal with the Software without restriction, including without limitation
C the rights to use, copy, modify, merge, publish, distribute, sublicense,
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C             1. Redistributions of source code must retain the above copyright
C               notice, this list of conditions and the following disclaimers.
C            2. Redistributions in binary form must reproduce the above 
C               copyright notice, this list of conditions and the following 
C               disclaimers in the documentation and/or other materials 
C               provided with the distribution.
C            3. Neither the names of Large Scale Research Systems Laboratory,
C               University of Illinois, nor the names of its contributors may
C               be used to endorse or promote products derived from this 
C               Software without specific prior written permission.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
C OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
C ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
C DEALINGS IN THE SOFTWARE.
C----------------------------------------------------------------------
C
C  Users should cite the original code using the following references:
C
C    J.J. Downs and E.F. Vogel, "A plant-wide industrial process control 
C    problem." Presented at the AIChE 1990 Annual Meeting, Session on
C    Industrial Challenge Problems in Process Control, Paper #24a
C    Chicago, Illinois, November 14, 1990.
c
C    J.J. Downs and E.F. Vogel, "A plant-wide industrial process control 
C    problem," Computers and Chemical Engineering, 17:245-255 (1993).
C  
C  Users should cite the modified code using the following references:
C
C    E.L. Russell, L.H. Chiang, and R.D. Braatz. Data-driven Techniques 
C    for Fault Detection and Diagnosis in Chemical Processes, Springer-Verlag, 
C    London, 2000. 
C
C    L.H. Chiang, E.L. Russell, and R.D. Braatz. Fault Detection and 
C    Diagnosis in Industrial Systems, Springer-Verlag, London, 2001.  
C
C    L.H. Chiang, E.L. Russell, and R.D. Braatz. "Fault diagnosis in 
C    chemical processes using Fisher discriminant analysis, discriminant 
C    partial least squares, and principal component analysis," Chemometrics 
C    and Intelligent Laboratory Systems, 50:243-252, 2000. 
C
C    E.L. Russell, L.H. Chiang, and R.D. Braatz. "Fault detection in 
C    industrial processes using canonical variate analysis and dynamic 
C    principal component analysis," Chemometrics and Intelligent Laboratory 
C    Systems, 51:81-93, 2000. 
C
C
C  Main program for demonstrating application of the modified Tennessee Eastman
C  Process Control Test Problem
C
      SUBROUTINE tesub(RANDOMSEED,NPTS,TSPERSAMPLE,
     +        numsamples,numfeat,numclass,X,Y,numevents,colevent,EVENTS)
      IMPLICIT NONE
      DOUBLE PRECISION RANDOMSEED
      integer NPTS,TSPERSAMPLE
      integer numsamples,numfeat,numclass,numevents,colevent
      double precision X(numsamples,numfeat), Y(numsamples,numclass)
      integer EVENTS(numevents,colevent)   ! Step fault on=1/off=0
C  
C      INTEGER IDV_(20)   ! Multiple fault vector
C                  1 2 3 4 5 6 7 8 9 1011121314151617181920
C      DATA IDV_  /0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0/     ! ORIGINAL
C      DATA IDV_  /0,1,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0,0,0/
C
C
C  MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   DISTURBANCE VECTOR COMMON BLOCK
C
      INTEGER IDV
      COMMON/DVEC/ IDV(20)
C
C   CONTROLLER COMMON BLOCK
C
C      DOUBLE PRECISION SETPT, GAIN, TAUI, ERROLD, DELTAT
C      COMMON/CTRL/ SETPT, GAIN, TAUI, ERROLD, DELTAT
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      INTEGER FLAG
      COMMON/FLAG6/ FLAG
C
      DOUBLE PRECISION GAIN1, ERROLD1
      COMMON/CTRL1/ GAIN1, ERROLD1
      DOUBLE PRECISION GAIN2, ERROLD2
      COMMON/CTRL2/ GAIN2, ERROLD2
      DOUBLE PRECISION GAIN3, ERROLD3
      COMMON/CTRL3/ GAIN3, ERROLD3
      DOUBLE PRECISION  GAIN4, ERROLD4
      COMMON/CTRL4/ GAIN4, ERROLD4
      DOUBLE PRECISION GAIN5, TAUI5, ERROLD5
      COMMON/CTRL5/ GAIN5, TAUI5, ERROLD5
      DOUBLE PRECISION GAIN6, ERROLD6
      COMMON/CTRL6/ GAIN6, ERROLD6
      DOUBLE PRECISION GAIN7, ERROLD7
      COMMON/CTRL7/  GAIN7, ERROLD7
      DOUBLE PRECISION GAIN8, ERROLD8
      COMMON/CTRL8/ GAIN8, ERROLD8
      DOUBLE PRECISION GAIN9, ERROLD9
      COMMON/CTRL9/ GAIN9, ERROLD9
      DOUBLE PRECISION GAIN10, TAUI10, ERROLD10
      COMMON/CTRL10/ GAIN10, TAUI10, ERROLD10
      DOUBLE PRECISION GAIN11, TAUI11, ERROLD11
      COMMON/CTRL11/ GAIN11, TAUI11, ERROLD11
      DOUBLE PRECISION GAIN13, TAUI13, ERROLD13
      COMMON/CTRL13/ GAIN13, TAUI13, ERROLD13
      DOUBLE PRECISION GAIN14, TAUI14, ERROLD14
      COMMON/CTRL14/ GAIN14, TAUI14, ERROLD14
      DOUBLE PRECISION GAIN15, TAUI15, ERROLD15
      COMMON/CTRL15/ GAIN15, TAUI15, ERROLD15
      DOUBLE PRECISION GAIN16, TAUI16, ERROLD16
      COMMON/CTRL16/ GAIN16, TAUI16, ERROLD16
      DOUBLE PRECISION GAIN17, TAUI17, ERROLD17
      COMMON/CTRL17/ GAIN17, TAUI17, ERROLD17
      DOUBLE PRECISION GAIN18, TAUI18, ERROLD18
      COMMON/CTRL18/ GAIN18, TAUI18, ERROLD18
      DOUBLE PRECISION GAIN19, TAUI19, ERROLD19
      COMMON/CTRL19/ GAIN19, TAUI19, ERROLD19
      DOUBLE PRECISION GAIN20, TAUI20, ERROLD20
      COMMON/CTRL20/ GAIN20, TAUI20, ERROLD20
      DOUBLE PRECISION GAIN22, TAUI22, ERROLD22
      COMMON/CTRL22/ GAIN22, TAUI22, ERROLD22
C
C  Local Variables
C
      INTEGER I, J, sample, F, NN, SSPTS, TEST, TEST1, TEST3, TEST4
      character*80 strbuf
C
      DOUBLE PRECISION TIME, YY(50), YP(50)
C      INTEGER TIMESTAMP(MAXEVENTS)
      INTEGER nexteventcnt, eventcnt
      INTEGER fault, state
C
C
C  Set the number of differential equations (states).  The process has 50
C  states.  If the user wishes to integrate additional states, NN must be
C  increased by the number of additional differential equations.
C
      NN = 50
C
C  Set the number of pints to simulate in steady state operation
C

      SSPTS = 0
C
C  Integrator Step Size:  1 Second Converted to Hours
C
      DELTAT = 1. / 3600.
C
      sample = 1
C  Initialize Process
C  (Sets TIME to zero)
C
      CALL TEINIT(NN,TIME,YY,YP,RANDOMSEED)
C
C  Set Controller Parameters
C  Make a Stripper Level Set Point Change of +15%
C
CC      SETPT = XMEAS(15) + 15.0
CC      GAIN = 2.0
CC      TAUI = 5.0
CC      ERROLD = 0.0
      SETPT(1)=3664.0        
      GAIN1=1.0
      ERROLD1=0.0
      SETPT(2)=4509.3
      GAIN2=1.0
      ERROLD2=0.0
      SETPT(3)=.25052
      GAIN3=1.
      ERROLD3=0.0
      SETPT(4)=9.3477
      GAIN4=1.
      ERROLD4=0.0
      SETPT(5)=26.902
      GAIN5=-0.083          
      TAUI5=1./3600.   
      ERROLD5=0.0
      SETPT(6)=0.33712  
      GAIN6=1.22                     
      ERROLD6=0.0
      SETPT(7)=50.0
      GAIN7=-2.06      
      ERROLD7=0.0
      SETPT(8)=50.0
      GAIN8=-1.62      
      ERROLD8=0.0
      SETPT(9)=230.31
      GAIN9=0.41          
      ERROLD9=0.0      
      SETPT(10)=94.599
      GAIN10= -0.156     * 10.
      TAUI10=1452./3600. 
      ERROLD10=0.0
      SETPT(11)=22.949    
      GAIN11=1.09        
      TAUI11=2600./3600.
      ERROLD11=0.0
      SETPT(13)=32.188
      GAIN13=18.              
      TAUI13=3168./3600.   
      ERROLD13=0.0
      SETPT(14)=6.8820
      GAIN14=8.3        
      TAUI14=3168.0/3600.
      ERROLD14=0.0
      SETPT(15)=18.776                     
      GAIN15=2.37              
      TAUI15=5069./3600.    
      ERROLD15=0.0
      SETPT(16)=65.731
      GAIN16=1.69        / 10.
      TAUI16=236./3600.
      ERROLD16=0.0
      SETPT(17)=75.000
      GAIN17=11.1      / 10.
      TAUI17=3168./3600.  
      ERROLD17=0.0        
      SETPT(18)=120.40
      GAIN18=2.83      * 10.
      TAUI18=982./3600.
      ERROLD18=0.0
      SETPT(19)=13.823
      GAIN19=-83.2        / 5. /3.  
      TAUI19=6336./3600. 
      ERROLD19=0.0
      SETPT(20)=0.83570  
      GAIN20=-16.3       / 5.         
      TAUI20=12408./3600.  
      ERROLD20=0.0
      SETPT(12)=2633.7
      GAIN22=-1.0        * 5.         
      TAUI22=1000./3600.  
      ERROLD22=0.0
C
C    Example Disturbance:
C    Change Reactor Cooling
C
      XMV(1) = 63.053 + 0.
      XMV(2) = 53.980 + 0.
      XMV(3) = 24.644 + 0.    
      XMV(4) = 61.302 + 0.
      XMV(5) = 22.210 + 0.
      XMV(6) = 40.064 + 0.
      XMV(7) = 38.100 + 0.
      XMV(8) = 46.534 + 0.
      XMV(9) = 47.446 + 0.
      XMV(10)= 41.106 + 0.
      XMV(11)= 18.114 + 0.
C
C      SETPT(6)=SETPT(6) + 0.2
C
C  Set all Disturbance Flags to OFF
C
      DO 100 I = 1, numclass
          IDV(I) = 0
 100  CONTINUE
C
C
C  Simulation Loop
C
      write(strbuf,'(1X,A)')'Starting simulation...\n'
      call teinfo(strbuf)
      do i=1,eventcnt
         write(strbuf,'(1x,A,I4,A,I8,A,I3,A,I3,A)') 'Timestamp ', i,
     +   ' =', EVENTS(i,3),
     +   ' fault=',events(i,2),' state=',events(i,3),'\n'
      end do
      call teinfo(strbuf)


      eventcnt = -1
      nexteventcnt = -1
      if( numevents .ge. 1 ) then
          eventcnt = 1
          nexteventcnt = EVENTS(eventcnt,1)
      endif

      DO 1000 I = 1, NPTS
        TEST3=MOD(I,5000)       
        if (TEST3.EQ.0) then
          write(strbuf,'(1x,A16,I8,A4,I8,A28,A13,A)')
     +                 'Simulation step ',i,' of ',NPTS,
     + '\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b',
     + '\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b'
          call teinfo(strbuf)
        endif

C Check if there is an event happens at time i
          do while( nexteventcnt .eq. I )
C             write(*,*) 'nexteventcnt=',nexteventcnt,'i=',i
              if( nexteventcnt .ne. EVENTS(eventcnt,1) ) then
                  write(strbuf,'(1X,A,I10,A,I10,A)')
     +             'Invalid time stamp',nexteventcnt,
     +             ' <>  ', EVENTS(eventcnt,1), '\n'
                  stop
              else
                 fault = EVENTS(eventcnt,2)
                 state = EVENTS(eventcnt,3)
                 IDV(fault) = state
                 eventcnt = eventcnt + 1
                 if( eventcnt .gt. numevents ) then
                     nexteventcnt = -1
                 else
                     nexteventcnt = EVENTS(eventcnt,1)
                 endif
                 write(strbuf,'(1X,A,I8,A,I7,A,I5,A,I5,A,I8,A)')
     +             'Event at time ',i,' fault=',fault,' state=',state,
     +             ' ts#=',eventcnt-1,' nextts=',nexteventcnt,'\n'
                 call mexPrintf(strbuf)
              endif
           enddo

      TEST=MOD(I,3)
      IF (TEST.EQ.0) THEN
            CALL CONTRL1
            CALL CONTRL2
            CALL CONTRL3
            CALL CONTRL4
            CALL CONTRL5
            CALL CONTRL6
            CALL CONTRL7
            CALL CONTRL8
            CALL CONTRL9
            CALL CONTRL10
            CALL CONTRL11
            CALL CONTRL16
            CALL CONTRL17
            CALL CONTRL18
        ENDIF
        TEST1=MOD(I,360)
        IF (TEST1.EQ.0) THEN
            CALL CONTRL13
            CALL CONTRL14
            CALL CONTRL15
            CALL CONTRL19
        ENDIF
        TEST1=MOD(I,900)
        IF (TEST1.EQ.0) CALL CONTRL20
C        TEST3=MOD(I,5000)       
C        IF (TEST3.EQ.0) THEN
C            write(strbuf,'(1X,A31,I8,A)')
C     +         '\nSimulation time (in seconds) =',I,'\n'
C            call teinfo(strbuf)
C        ENDIF
C
      TEST4=MOD(I,TSPERSAMPLE)      
      IF (TEST4.EQ.0) THEN
C Copy into feature vector X and class vector Y
          do j=1,12
            X(sample,j) = XMV(j);
          enddo
          do j=1,41
            X(sample,j+12) = XMEAS(j)
          enddo
          do j=1,numclass
            Y(sample,j) = IDV(j)
          enddo
          sample = sample + 1;
      ENDIF
C
      CALL INTGTR(NN,TIME,DELTAT,YY,YP)
C
      CALL CONSHAND
C
 1000 CONTINUE
      write(strbuf,'(1X,A)')'\nSimulation is done.\n'
      call teinfo(strbuf)

      END
C
C=============================================================================
C
CC      SUBROUTINE CONTRL
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
CC      DOUBLE PRECISION XMEAS, XMV
CC      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
CC      DOUBLE PRECISION SETPT, GAIN, TAUI, ERROLD, DELTAT
CC      COMMON/CTRL/ SETPT, GAIN, TAUI, ERROLD, DELTAT
C
CC      DOUBLE PRECISION ERR, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
CC      ERR = SETPT - XMEAS(15)
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
CC      DXMV = GAIN * ( ( ERR - ERROLD ) + ERR * DELTAT * 60. / TAUI )
C
CC      XMV(8) = XMV(8) - DXMV
C
CC      ERROLD = ERR
C
CC      RETURN
CC      END
C
C=============================================================================
C
      SUBROUTINE CONTRL1
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN1, ERROLD1
      COMMON/CTRL1/ GAIN1, ERROLD1
C
      DOUBLE PRECISION ERR1, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR1 = (SETPT(1) - XMEAS(2)) * 100. / 5811.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN1 * ( ( ERR1 - ERROLD1 ) )
C
      XMV(1) = XMV(1) + DXMV
C
      ERROLD1 = ERR1
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL2
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN2, ERROLD2
      COMMON/CTRL2/ GAIN2, ERROLD2
C
      DOUBLE PRECISION ERR2, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR2 = (SETPT(2) - XMEAS(3)) * 100. / 8354. 
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN2 * ( ( ERR2 - ERROLD2 ) )
C
      XMV(2) = XMV(2) + DXMV
C
      ERROLD2 = ERR2
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL3
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN3, ERROLD3
      COMMON/CTRL3/ GAIN3, ERROLD3
C
      DOUBLE PRECISION ERR3, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR3 = (SETPT(3) - XMEAS(1)) * 100. / 1.017
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN3 * ( ( ERR3 - ERROLD3 ) )
C
      XMV(3) = XMV(3) + DXMV
C
      ERROLD3 = ERR3
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL4
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN4, ERROLD4
      COMMON/CTRL4/ GAIN4, ERROLD4
C
      DOUBLE PRECISION ERR4, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR4 = (SETPT(4) - XMEAS(4)) * 100. / 15.25
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN4 * ( ( ERR4 - ERROLD4 ) )
C
      XMV(4) = XMV(4) + DXMV
C
      ERROLD4 = ERR4
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL5
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN5, TAUI5, ERROLD5
      COMMON/CTRL5/ GAIN5, TAUI5, ERROLD5
C
      DOUBLE PRECISION ERR5, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR5 = (SETPT(5) - XMEAS(5))  * 100. / 53.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
C       PRINT *, 'GAIN5= ', GAIN5
C      PRINT *, 'TAUI5= ', TAUI5
C      PRINT *, 'ERR5= ', ERR5
C      PRINT *, 'ERROLD5= ', ERROLD5     
C
      DXMV = GAIN5 * ((ERR5 - ERROLD5)+ERR5*DELTAT*3./TAUI5)
C
      XMV(5) = XMV(5) + DXMV
C
      ERROLD5 = ERR5
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL6
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
      INTEGER FLAG
       COMMON/FLAG6/ FLAG
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN6, ERROLD6
      COMMON/CTRL6/ GAIN6, ERROLD6
C
      DOUBLE PRECISION ERR6, DXMV
C
C  Example PI Controller:
C     Stripper Level Controller
      IF (XMEAS(13).GE.2950.0) THEN
            XMV(6)=100.0
            FLAG=1
      ELSEIF (FLAG.EQ.1.AND.XMEAS(13).GE.2633.7) THEN
            XMV(6)=100.0
      ELSEIF (FLAG.EQ.1.AND.XMEAS(13).LE.2633.7) THEN
            XMV(6)=40.060
            SETPT(6)=0.33712
            ERROLD6=0.0
             FLAG=0
      ELSEIF (XMEAS(13).LE.2300.) THEN
            XMV(6)=0.0
            FLAG=2
      ELSEIF (FLAG.EQ.2.AND.XMEAS(13).LE.2633.7) THEN
            XMV(6)=0.0
      ELSEIF (FLAG.EQ.2.AND.XMEAS(13).GE.2633.7) THEN
            XMV(6)=40.060
            SETPT(6)=0.33712
            ERROLD6=0.0
            FLAG=0
      ELSE      
            FLAG=0
C
C    Calculate Error
C
       ERR6 = (SETPT(6) - XMEAS(10)) * 100. /1.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
C      PRINT *, 'XMV(6)= ', XMV(6)
      DXMV = GAIN6 * ( ( ERR6 - ERROLD6 ) )
C
C       PRINT *, 'GAIN6= ', GAIN6
C      PRINT *, 'SETPT(6)= ', SETPT(6)      
C      PRINT *, 'XMEAS(10)= ', XMEAS(10)     
      XMV(6) = XMV(6) + DXMV
C
C       PRINT *, 'ERROLD6= ', ERROLD6     
C      PRINT *, 'ERR6= ', ERR6
C      PRINT *, 'XMV(6)== ', XMV(6)
      ERROLD6 = ERR6
      ENDIF
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL7
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN7, ERROLD7
      COMMON/CTRL7/ GAIN7, ERROLD7
C
      DOUBLE PRECISION ERR7, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR7 = (SETPT(7) - XMEAS(12)) * 100. / 70.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN7 * ( ( ERR7 - ERROLD7 ) )
C
      XMV(7) = XMV(7) + DXMV
C
      ERROLD7 = ERR7
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL8
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN8, ERROLD8
      COMMON/CTRL8/ GAIN8, ERROLD8
C
      DOUBLE PRECISION ERR8, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR8 = (SETPT(8) - XMEAS(15)) * 100. / 70.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV =  GAIN8 * ( ( ERR8 - ERROLD8 ) )
C
      XMV(8) = XMV(8) + DXMV
C
      ERROLD8 = ERR8
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL9
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN9, ERROLD9
      COMMON/CTRL9/ GAIN9, ERROLD9
C
      DOUBLE PRECISION ERR9, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR9 = (SETPT(9) - XMEAS(19)) * 100. / 460. 
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN9 * ( ( ERR9 - ERROLD9 ) )
C
      XMV(9) = XMV(9) + DXMV
C
      ERROLD9 = ERR9
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL10
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN10, TAUI10, ERROLD10
      COMMON/CTRL10/ GAIN10, TAUI10, ERROLD10
C
      DOUBLE PRECISION ERR10, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR10 = (SETPT(10) - XMEAS(21)) * 100. / 150.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN10*((ERR10 - ERROLD10)+ERR10*DELTAT*3./TAUI10)
C
      XMV(10) = XMV(10) + DXMV
C
      ERROLD10 = ERR10
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL11
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN11, TAUI11, ERROLD11
      COMMON/CTRL11/ GAIN11, TAUI11, ERROLD11
C
      DOUBLE PRECISION ERR11, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR11 = (SETPT(11) - XMEAS(17)) * 100. / 46.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN11*((ERR11 - ERROLD11)+ERR11*DELTAT*3./TAUI11)
C
      XMV(11) = XMV(11) + DXMV
C
      ERROLD11 = ERR11
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL13
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN13, TAUI13, ERROLD13
      COMMON/CTRL13/ GAIN13, TAUI13, ERROLD13
C
      DOUBLE PRECISION ERR13, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR13 = (SETPT(13) - XMEAS(23)) * 100. / 100.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN13 * ((ERR13 - ERROLD13)+ERR13*DELTAT*360./TAUI13)
C
      SETPT(3) = SETPT(3) + DXMV * 1.017 / 100.
C
      ERROLD13 = ERR13
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL14
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN14, TAUI14, ERROLD14
      COMMON/CTRL14/ GAIN14, TAUI14, ERROLD14
C
      DOUBLE PRECISION ERR14, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR14 = (SETPT(14) - XMEAS(26)) * 100. /100.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN14*((ERR14 - ERROLD14)+ERR14*DELTAT*360./TAUI14)
C
      SETPT(1) = SETPT(1) + DXMV * 5811. / 100.
C
      ERROLD14 = ERR14
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL15
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN15, TAUI15, ERROLD15
      COMMON/CTRL15/ GAIN15, TAUI15, ERROLD15
C
      DOUBLE PRECISION ERR15, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR15 = (SETPT(15) - XMEAS(27)) * 100. / 100.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN15 * ((ERR15 - ERROLD15)+ERR15*DELTAT*360./TAUI15)
C
      SETPT(2) = SETPT(2) + DXMV * 8354. / 100.
C
      ERROLD15 = ERR15
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL16
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN16, TAUI16, ERROLD16
      COMMON/CTRL16/ GAIN16, TAUI16, ERROLD16
C
      DOUBLE PRECISION ERR16, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR16 = (SETPT(16) - XMEAS(18)) * 100. / 130.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN16 * ((ERR16 - ERROLD16)+ERR16*DELTAT*3./TAUI16)
C
      SETPT(9) = SETPT(9) + DXMV * 460. / 100.
C
      ERROLD16 = ERR16
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL17
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN17, TAUI17, ERROLD17
      COMMON/CTRL17/ GAIN17, TAUI17, ERROLD17
C
      DOUBLE PRECISION ERR17, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR17 = (SETPT(17) - XMEAS(8)) * 100. / 50.
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV =GAIN17*((ERR17 - ERROLD17)+ERR17*DELTAT*3./TAUI17)
C
      SETPT(4) = SETPT(4) + DXMV * 15.25 / 100.
C
      ERROLD17 = ERR17
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL18
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN18, TAUI18, ERROLD18
      COMMON/CTRL18/ GAIN18, TAUI18, ERROLD18
C
      DOUBLE PRECISION ERR18, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR18 = (SETPT(18) - XMEAS(9)) * 100. / 150. 
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN18 * ((ERR18 - ERROLD18)+ERR18*DELTAT*3./TAUI18)
C
      SETPT(10) = SETPT(10) + DXMV * 150. / 100.
C
      ERROLD18 = ERR18
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL19
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN19, TAUI19, ERROLD19
      COMMON/CTRL19/ GAIN19, TAUI19, ERROLD19
C
      DOUBLE PRECISION ERR19, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR19 = (SETPT(19) - XMEAS(30)) * 100. / 26.
C      PRINT *, 'ERROLD19= ', ERROLD19
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN19*((ERR19 - ERROLD19)+ERR19*DELTAT*360./TAUI19)
C
      SETPT(6) = SETPT(6) + DXMV * 1. / 100.
C      PRINT *, 'SETPT(6)= ', SETPT(6)
C
      ERROLD19 = ERR19
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL20
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN20, TAUI20, ERROLD20
      COMMON/CTRL20/  GAIN20, TAUI20, ERROLD20
C    
      DOUBLE PRECISION ERR20, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR20 = (SETPT(20) - XMEAS(38)) * 100. / 1.6
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN20*((ERR20 - ERROLD20)+ERR20*DELTAT*900./TAUI20)
C
      SETPT(16) = SETPT(16) + DXMV  * 130. / 100.
C
      ERROLD20 = ERR20
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONTRL22
C
C  Discrete control algorithms
C
C
C   MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
C   CONTROLLER COMMON BLOCK
C
      DOUBLE PRECISION SETPT, DELTAT
      COMMON/CTRLALL/ SETPT(20), DELTAT
      DOUBLE PRECISION GAIN22, TAUI22, ERROLD22
      COMMON/CTRL22/  GAIN22, TAUI22, ERROLD22
C    
      DOUBLE PRECISION ERR22, DXMV
C
C  Example PI Controller:
C    Stripper Level Controller
C
C    Calculate Error
C
      ERR22 = SETPT(12) - XMEAS(13)
C
C    Proportional-Integral Controller (Velocity Form)
C         GAIN = Controller Gain
C         TAUI = Reset Time (min)
C
      DXMV = GAIN22*((ERR22 - ERROLD22)+ERR22*DELTAT*3./TAUI22)
C
      XMV(6) = XMV(6) + DXMV
C
      ERROLD22 = ERR22
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE INTGTR(NN,TIME,DELTAT,YY,YP)
C
C  Euler Integration Algorithm
C
C
      INTEGER I, NN
C
      DOUBLE PRECISION TIME, DELTAT, YY(NN), YP(NN)
C
      CALL TEFUNC(NN,TIME,YY,YP)
C
      TIME = TIME + DELTAT
C
      DO 100 I = 1, NN
C
          YY(I) = YY(I) + YP(I) * DELTAT 
C
 100  CONTINUE
C
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE CONSHAND
C
C  Euler Integration Algorithm
C
C
      DOUBLE PRECISION XMEAS, XMV
      COMMON/PV/ XMEAS(41), XMV(12)
C
      INTEGER I
C      
      DO 100 I=1, 11
          IF (XMV(I).LE.0.0) XMV(I)=0.
              IF (XMV(I).GE.100.0) XMV(I)=100.
 100  CONTINUE
C
      RETURN
      END
C
C               Tennessee Eastman Process Control Test Problem
C
C                    James J. Downs and Ernest F. Vogel
C
C                  Process and Control Systems Engineering
C                        Tennessee Eastman Company
C                              P.O. Box 511
C                          Kingsport,TN  37662
C
C  Reference:
C    "A Plant-Wide Industrial Process Control Problem"
C    Presented at the AIChE 1990 Annual Meeting
C    Industrial Challenge Problems in Process Control,Paper #24a
C    Chicago,Illinois,November 14,1990
C
C  Revised 4-4-91 to correct error in documentation of manipulated variables
C
C  Subroutines:
C
C    TEFUNC - Function evaluator to be called by integrator
C    TEINIT - Initialization
C    TESUBi - Utility subroutines, i=1,2,..,8
C
C
C  The process simulation has 50 states (NN=50).  If the user wishes to
C  integrate additional states, NN must be increased accordingly in the
C  calling program.  The additional states should be appended to the end
C  of the YY vector, e.g. YY(51),...  The additional derivatives should
C  be appended to the end of the YP vector, e.g. YP(51),...  To initialize
C  the new states and to calculate derivatives for them, we suggest
C  creating new function evaluator and initialization routines as follows.
C
C          C-----------------------------------------------
C          C
C                SUBROUTINE FUNC(NN,TIME,YY,YP)
C          C
C                INTEGER NN
C                DOUBLE PRECISION TIME, YY(NN), YP(NN)
C          C
C          C  Call the function evaluator for the process
C          C
C                CALL TEFUNC(NN,TIME,YY,YP)
C          C
C          C  Calculate derivatives for additional states
C          C
C                YP(51) = ....
C                YP(52) = ....
C                   .
C                   .
C                   .
C                YP(NN) = ....
C          C
C                RETURN
C                END
C          C
C          C-----------------------------------------------
C          C
C                SUBROUTINE INIT(NN,TIME,YY,YP)
C          C
C                INTEGER NN
C                DOUBLE PRECISION TIME, YY(NN), YP(NN)
C          C
C          C  Call the initialization for the process
C          C
C                CALL TEINIT(NN,TIME,YY,YP,USEORIGSEED)
C          C
C          C  Initialize additional states
C          C
C                YY(51) = ....
C                YY(52) = ....
C                   .
C                   .
C                   .
C                YY(NN) = ....
C          C
C                RETURN
C                END
C          C
C          C-----------------------------------------------
C
C  Differences between the code and its description in the paper:
C
C  1.  Subroutine TEINIT has TIME in the argument list.  TEINIT sets TIME
C      to zero.
C
C  2.  There are 8 utility subroutines (TESUBi) rather than 5.
C
C  3.  Process disturbances 14 through 20 do NOT need to be used in
C      conjunction with another disturbance as stated in the paper.  All
C      disturbances can be used alone or in any combination.
C
C
C  Manipulated Variables
C
C    XMV(1)     D Feed Flow (stream 2)            (Corrected Order)
C    XMV(2)     E Feed Flow (stream 3)            (Corrected Order)
C    XMV(3)     A Feed Flow (stream 1)            (Corrected Order)
C    XMV(4)     A and C Feed Flow (stream 4)
C    XMV(5)     Compressor Recycle Valve
C    XMV(6)     Purge Valve (stream 9)
C    XMV(7)     Separator Pot Liquid Flow (stream 10)
C    XMV(8)     Stripper Liquid Product Flow (stream 11)
C    XMV(9)     Stripper Steam Valve
C    XMV(10)    Reactor Cooling Water Flow
C    XMV(11)    Condenser Cooling Water Flow
C    XMV(12)    Agitator Speed
C
C  Continuous Process Measurements
C
C    XMEAS(1)   A Feed  (stream 1)                    kscmh
C    XMEAS(2)   D Feed  (stream 2)                    kg/hr
C    XMEAS(3)   E Feed  (stream 3)                    kg/hr
C    XMEAS(4)   A and C Feed  (stream 4)              kscmh
C    XMEAS(5)   Recycle Flow  (stream 8)              kscmh
C    XMEAS(6)   Reactor Feed Rate  (stream 6)         kscmh
C    XMEAS(7)   Reactor Pressure                      kPa gauge
C    XMEAS(8)   Reactor Level                         %
C    XMEAS(9)   Reactor Temperature                   Deg C
C    XMEAS(10)  Purge Rate (stream 9)                 kscmh
C    XMEAS(11)  Product Sep Temp                      Deg C
C    XMEAS(12)  Product Sep Level                     %
C    XMEAS(13)  Prod Sep Pressure                     kPa gauge
C    XMEAS(14)  Prod Sep Underflow (stream 10)        m3/hr
C    XMEAS(15)  Stripper Level                        %
C    XMEAS(16)  Stripper Pressure                     kPa gauge
C    XMEAS(17)  Stripper Underflow (stream 11)        m3/hr
C    XMEAS(18)  Stripper Temperature                  Deg C
C    XMEAS(19)  Stripper Steam Flow                   kg/hr
C    XMEAS(20)  Compressor Work                       kW
C    XMEAS(21)  Reactor Cooling Water Outlet Temp     Deg C
C    XMEAS(22)  Separator Cooling Water Outlet Temp   Deg C
C
C  Sampled Process Measurements
C
C    Reactor Feed Analysis (Stream 6)
C        Sampling Frequency = 0.1 hr
C        Dead Time = 0.1 hr
C        Mole %
C    XMEAS(23)   Component A
C    XMEAS(24)   Component B
C    XMEAS(25)   Component C
C    XMEAS(26)   Component D
C    XMEAS(27)   Component E
C    XMEAS(28)   Component F
C
C    Purge Gas Analysis (Stream 9)
C        Sampling Frequency = 0.1 hr
C        Dead Time = 0.1 hr
C        Mole %
C    XMEAS(29)   Component A
C    XMEAS(30)   Component B
C    XMEAS(31)   Component C
C    XMEAS(32)   Component D
C    XMEAS(33)   Component E
C    XMEAS(34)   Component F
C    XMEAS(35)   Component G
C    XMEAS(36)   Component H
C
C    Product Analysis (Stream 11)
C        Sampling Frequency = 0.25 hr
C        Dead Time = 0.25 hr
C        Mole %
C    XMEAS(37)   Component D
C    XMEAS(38)   Component E
C    XMEAS(39)   Component F
C    XMEAS(40)   Component G
C    XMEAS(41)   Component H
C
C  Process Disturbances
C
C    IDV(1)   A/C Feed Ratio, B Composition Constant (Stream 4)          Step
C    IDV(2)   B Composition, A/C Ratio Constant (Stream 4)               Step
C    IDV(3)   D Feed Temperature (Stream 2)                              Step
C    IDV(4)   Reactor Cooling Water Inlet Temperature                    Step
C    IDV(5)   Condenser Cooling Water Inlet Temperature                  Step
C    IDV(6)   A Feed Loss (Stream 1)                                     Step
C    IDV(7)   C Header Pressure Loss - Reduced Availability (Stream 4)   Step
C    IDV(8)   A, B, C Feed Composition (Stream 4)            Random Variation
C    IDV(9)   D Feed Temperature (Stream 2)                  Random Variation
C    IDV(10)  C Feed Temperature (Stream 4)                  Random Variation
C    IDV(11)  Reactor Cooling Water Inlet Temperature        Random Variation
C    IDV(12)  Condenser Cooling Water Inlet Temperature      Random Variation
C    IDV(13)  Reaction Kinetics                                    Slow Drift
C    IDV(14)  Reactor Cooling Water Valve                            Sticking
C    IDV(15)  Condenser Cooling Water Valve                          Sticking
C    IDV(16)  Unknown
C    IDV(17)  Unknown
C    IDV(18)  Unknown
C    IDV(19)  Unknown
C    IDV(20)  Unknown
C
C
C=============================================================================
C
      SUBROUTINE TEFUNC(NN,TIME,YY,YP)
      IMPLICIT NONE
C
C       Function Evaluator
C
C         Inputs:
C
C           NN   = Number of differential equations
C           Time = Current time(hrs)
C           YY   = Current state values
C
C         Outputs:
C
C           YP   = Current derivative values
C
      DOUBLE PRECISION XMEAS,XMV
      COMMON/PV/XMEAS(41),XMV(12)
      INTEGER IDV
      COMMON/DVEC/IDV(20)
      DOUBLE PRECISION
     .UCLR,UCVR,UTLR,UTVR,
     .XLR,XVR,ETR,ESR,
     .TCR,TKR,DLR,
     .VLR,VVR,VTR,
     .PTR,PPR,
     .CRXR,RR,RH,
     .FWR,TWR,QUR,HWR,UAR,
     .UCLS,UCVS,UTLS,UTVS,
     .XLS,XVS,ETS,ESS,
     .TCS,TKS,DLS,
     .VLS,VVS,VTS,
     .PTS,PPS,
     .FWS,TWS,QUS,HWS,
     .UCLC,UTLC,XLC,
     .ETC,ESC,TCC,DLC,
     .VLC,VTC,QUC,
     .UCVV,UTVV,XVV,
     .ETV,ESV,TCV,TKV,
     .VTV,PTV,
     .VCV,VRNG,VTAU,
     .FTM,
     .FCM,XST,XMWS,
     .HST,TST,SFR,
     .CPFLMX,CPPRMX,CPDH,
     .TCWR,TCWS,
     .HTR,AGSP,
     .XDEL,XNS,
     .TGAS,TPROD,VST
      INTEGER
     .IVST
      COMMON/TEPROC/
     .UCLR(8),UCVR(8),UTLR,UTVR,
     .XLR(8),XVR(8),ETR,ESR,
     .TCR,TKR,DLR,
     .VLR,VVR,VTR,
     .PTR,PPR(8),
     .CRXR(8),RR(4),RH,
     .FWR,TWR,QUR,HWR,UAR,
     .UCLS(8),UCVS(8),UTLS,UTVS,
     .XLS(8),XVS(8),ETS,ESS,
     .TCS,TKS,DLS,
     .VLS,VVS,VTS,
     .PTS,PPS(8),
     .FWS,TWS,QUS,HWS,
     .UCLC(8),UTLC,XLC(8),
     .ETC,ESC,TCC,DLC,
     .VLC,VTC,QUC,
     .UCVV(8),UTVV,XVV(8),
     .ETV,ESV,TCV,TKV,
     .VTV,PTV,
     .VCV(12),VRNG(12),VTAU(12),
     .FTM(13),
     .FCM(8,13),XST(8,13),XMWS(13),
     .HST(13),TST(13),SFR(8),
     .CPFLMX,CPPRMX,CPDH,
     .TCWR,TCWS,
     .HTR(3),AGSP,
     .XDEL(41),XNS(41),
     .TGAS,TPROD,VST(12),IVST(12)
      INTEGER IDVWLK
      DOUBLE PRECISION
     .ADIST,
     .BDIST,
     .CDIST,
     .DDIST,
     .TLAST,
     .TNEXT,
     .HSPAN,
     .HZERO,
     .SSPAN,
     .SZERO,
     .SPSPAN
      COMMON/WLK/
     .ADIST(12),
     .BDIST(12),
     .CDIST(12),
     .DDIST(12),
     .TLAST(12),
     .TNEXT(12),
     .HSPAN(12),
     .HZERO(12),
     .SSPAN(12),
     .SZERO(12),
     .SPSPAN(12),
     .IDVWLK(12)
      DOUBLE PRECISION
     .AVP,BVP,CVP,
     .AH,BH,CH,
     .AG,BG,CG,
     .AV,
     .AD,BD,CD,
     .XMW
      COMMON/CONST/
     .AVP(8),BVP(8),CVP(8),
     .AH(8),BH(8),CH(8),
     .AG(8),BG(8),CG(8),
     .AV(8),
     .AD(8),BD(8),CD(8),
     .XMW(8)
      INTEGER NN,I,ISD
      DOUBLE PRECISION RG,
     .VPR,
     .FIN(8),
     .YY(NN),
     .YP(NN),
     .TIME,
     .FLMS,
     .DLP,
     .PR,
     .FLCOEF,
     .UAS,
     .UAC,
     .VOVRL,
     .UARLEV,
     .VPOS(12),
     .XMNS,
     .XCMP(41),
     .TMPFAC,
     .R1F,
     .R2F,
     .HWLK,
     .SWLK,
     .SPWLK,
     .TESUB7,
     .TESUB8
      DO 500 I=1,20
      IF(IDV(I).GT.0)THEN
      IDV(I)=1
      ELSE
      IDV(I)=0
      ENDIF
 500  CONTINUE
      IDVWLK(1)=IDV(8)
      IDVWLK(2)=IDV(8)
      IDVWLK(3)=IDV(9)
      IDVWLK(4)=IDV(10)
      IDVWLK(5)=IDV(11)
      IDVWLK(6)=IDV(12)
      IDVWLK(7)=IDV(13)
      IDVWLK(8)=IDV(13)
      IDVWLK(9)=IDV(16)
      IDVWLK(10)=IDV(17)
      IDVWLK(11)=IDV(18)
      IDVWLK(12)=IDV(20)
      DO 900 I=1,9
      IF(TIME.GE.TNEXT(I))THEN
      HWLK=TNEXT(I)-TLAST(I)
      SWLK=ADIST(I)+HWLK*(BDIST(I)+HWLK*
     .(CDIST(I)+HWLK*DDIST(I)))
      SPWLK=BDIST(I)+HWLK*
     .(2.D0*CDIST(I)+3.D0*HWLK*DDIST(I))
      TLAST(I)=TNEXT(I)
      CALL TESUB5(SWLK,SPWLK,ADIST(I),BDIST(I),CDIST(I),
     .DDIST(I),TLAST(I),TNEXT(I),HSPAN(I),HZERO(I),
     .SSPAN(I),SZERO(I),SPSPAN(I),IDVWLK(I))
      ENDIF
  900 CONTINUE
      DO 910 I=10,12
      IF(TIME.GE.TNEXT(I))THEN
      HWLK=TNEXT(I)-TLAST(I)
      SWLK=ADIST(I)+HWLK*(BDIST(I)+HWLK*
     .(CDIST(I)+HWLK*DDIST(I)))
      SPWLK=BDIST(I)+HWLK*
     .(2.D0*CDIST(I)+3.D0*HWLK*DDIST(I))
      TLAST(I)=TNEXT(I)
      IF(SWLK.GT.0.1D0)THEN
      ADIST(I)=SWLK
      BDIST(I)=SPWLK
      CDIST(I)=-(3.D0*SWLK+0.2D0*SPWLK)/0.01D0
      DDIST(I)=(2.D0*SWLK+0.1D0*SPWLK)/0.001D0
      TNEXT(I)=TLAST(I)+0.1D0
      ELSE
      ISD=-1
      HWLK=HSPAN(I)*TESUB7(ISD)+HZERO(I)
      ADIST(I)=0.D0
      BDIST(I)=0.D0
      CDIST(I)=DBLE(IDVWLK(I))/HWLK**2
      DDIST(I)=0.D0
      TNEXT(I)=TLAST(I)+HWLK
      ENDIF
      ENDIF
  910 CONTINUE
      IF(TIME.EQ.0.D0)THEN
      DO 950 I=1,12
      ADIST(I)=SZERO(I)
      BDIST(I)=0.D0
      CDIST(I)=0.D0
      DDIST(I)=0.D0
      TLAST(I)=0.0D0
      TNEXT(I)=0.1D0
  950 CONTINUE
      END IF
      XST(1,4)=TESUB8(1,TIME)-IDV(1)*0.03D0
     .-IDV(2)*2.43719D-3
      XST(2,4)=TESUB8(2,TIME)+IDV(2)*0.005D0
      XST(3,4)=1.D0-XST(1,4)-XST(2,4)
      TST(1)=TESUB8(3,TIME)+IDV(3)*5.D0
      TST(4)=TESUB8(4,TIME)
      TCWR=TESUB8(5,TIME)+IDV(4)*5.D0
      TCWS=TESUB8(6,TIME)+IDV(5)*5.D0
      R1F=TESUB8(7,TIME)
      R2F=TESUB8(8,TIME)
      DO 1010 I=1,3
      UCVR(I)=YY(I)
      UCVS(I)=YY(I+9)
      UCLR(I)=0.0
      UCLS(I)=0.0
 1010 CONTINUE
      DO 1020 I=4,8
      UCLR(I)=YY(I)
      UCLS(I)=YY(I+9)
 1020 CONTINUE
      DO 1030 I=1,8
      UCLC(I)=YY(I+18)
      UCVV(I)=YY(I+27)
 1030 CONTINUE
      ETR=YY(9)
      ETS=YY(18)
      ETC=YY(27)
      ETV=YY(36)
      TWR=YY(37)
      TWS=YY(38)
      DO 1035 I=1,12
      VPOS(I)=YY(I+38)
 1035 CONTINUE
      UTLR=0.0
      UTLS=0.0
      UTLC=0.0
      UTVV=0.0
      DO 1040 I=1,8
      UTLR=UTLR+UCLR(I)
      UTLS=UTLS+UCLS(I)
      UTLC=UTLC+UCLC(I)
      UTVV=UTVV+UCVV(I)
 1040 CONTINUE
      DO 1050 I=1,8
      XLR(I)=UCLR(I)/UTLR
      XLS(I)=UCLS(I)/UTLS
      XLC(I)=UCLC(I)/UTLC
      XVV(I)=UCVV(I)/UTVV
 1050 CONTINUE
      ESR=ETR/UTLR
      ESS=ETS/UTLS
      ESC=ETC/UTLC
      ESV=ETV/UTVV
      CALL TESUB2(XLR,TCR,ESR,0)
      TKR=TCR+273.15
      CALL TESUB2(XLS,TCS,ESS,0)
      TKS=TCS+273.15
      CALL TESUB2(XLC,TCC,ESC,0)
      CALL TESUB2(XVV,TCV,ESV,2)
      TKV=TCV+273.15
      CALL TESUB4(XLR,TCR,DLR)
      CALL TESUB4(XLS,TCS,DLS)
      CALL TESUB4(XLC,TCC,DLC)
      VLR=UTLR/DLR
      VLS=UTLS/DLS
      VLC=UTLC/DLC
      VVR=VTR-VLR
      VVS=VTS-VLS
      RG=998.9
      PTR=0.0
      PTS=0.0
      DO 1110 I=1,3
      PPR(I)=UCVR(I)*RG*TKR/VVR
      PTR=PTR+PPR(I)
      PPS(I)=UCVS(I)*RG*TKS/VVS
      PTS=PTS+PPS(I)
 1110 CONTINUE
      DO 1120 I=4,8
      VPR=DEXP(AVP(I)+BVP(I)/(TCR+CVP(I)))
      PPR(I)=VPR*XLR(I)
      PTR=PTR+PPR(I)
      VPR=DEXP(AVP(I)+BVP(I)/(TCS+CVP(I)))
      PPS(I)=VPR*XLS(I)
      PTS=PTS+PPS(I)
 1120 CONTINUE
      PTV=UTVV*RG*TKV/VTV
      DO 1130 I=1,8
      XVR(I)=PPR(I)/PTR
      XVS(I)=PPS(I)/PTS
 1130 CONTINUE
      UTVR=PTR*VVR/RG/TKR
      UTVS=PTS*VVS/RG/TKS
      DO 1140 I=4,8
      UCVR(I)=UTVR*XVR(I)
      UCVS(I)=UTVS*XVS(I)
 1140 CONTINUE
      RR(1)=DEXP(31.5859536-40000.0/1.987/TKR)*R1F
      RR(2)=DEXP(3.00094014-20000.0/1.987/TKR)*R2F
      RR(3)=DEXP(53.4060443-60000.0/1.987/TKR)
      RR(4)=RR(3)*0.767488334D0
      IF(PPR(1).GT.0.0.AND.PPR(3).GT.0.0)THEN
      R1F=PPR(1)**1.1544
      R2F=PPR(3)**0.3735
      RR(1)=RR(1)*R1F*R2F*PPR(4)
      RR(2)=RR(2)*R1F*R2F*PPR(5)
      ELSE
      RR(1)=0.0
      RR(2)=0.0
      ENDIF
      RR(3)=RR(3)*PPR(1)*PPR(5)
      RR(4)=RR(4)*PPR(1)*PPR(4)
      DO 1200 I=1,4
      RR(I)=RR(I)*VVR
 1200 CONTINUE
      CRXR(1)=-RR(1)-RR(2)-RR(3)
      CRXR(3)=-RR(1)-RR(2)
      CRXR(4)=-RR(1)-1.5D0*RR(4)
      CRXR(5)=-RR(2)-RR(3)
      CRXR(6)=RR(3)+RR(4)
      CRXR(7)=RR(1)
      CRXR(8)=RR(2)
      RH=RR(1)*HTR(1)+RR(2)*HTR(2)
      XMWS(1)=0.0
      XMWS(2)=0.0
      XMWS(6)=0.0
      XMWS(8)=0.0
      XMWS(9)=0.0
      XMWS(10)=0.0
      DO 2010 I=1,8
      XST(I,6)=XVV(I)
      XST(I,8)=XVR(I)
      XST(I,9)=XVS(I)
      XST(I,10)=XVS(I)
      XST(I,11)=XLS(I)
      XST(I,13)=XLC(I)
      XMWS(1)=XMWS(1)+XST(I,1)*XMW(I)
      XMWS(2)=XMWS(2)+XST(I,2)*XMW(I)
      XMWS(6)=XMWS(6)+XST(I,6)*XMW(I)
      XMWS(8)=XMWS(8)+XST(I,8)*XMW(I)
      XMWS(9)=XMWS(9)+XST(I,9)*XMW(I)
      XMWS(10)=XMWS(10)+XST(I,10)*XMW(I)
 2010 CONTINUE
      TST(6)=TCV
      TST(8)=TCR
      TST(9)=TCS
      TST(10)=TCS
      TST(11)=TCS
      TST(13)=TCC
      CALL TESUB1(XST(1,1),TST(1),HST(1),1)
      CALL TESUB1(XST(1,2),TST(2),HST(2),1)
      CALL TESUB1(XST(1,3),TST(3),HST(3),1)
      CALL TESUB1(XST(1,4),TST(4),HST(4),1)
      CALL TESUB1(XST(1,6),TST(6),HST(6),1)
      CALL TESUB1(XST(1,8),TST(8),HST(8),1)
      CALL TESUB1(XST(1,9),TST(9),HST(9),1)
      HST(10)=HST(9)
      CALL TESUB1(XST(1,11),TST(11),HST(11),0)
      CALL TESUB1(XST(1,13),TST(13),HST(13),0)
      FTM(1)=VPOS(1)*VRNG(1)/100.0
      FTM(2)=VPOS(2)*VRNG(2)/100.0
      FTM(3)=VPOS(3)*(1.D0-IDV(6))*VRNG(3)/100.0
      FTM(4)=VPOS(4)*(1.D0-IDV(7)*0.2D0)
     .*VRNG(4)/100.0+1.D-10
      FTM(11)=VPOS(7)*VRNG(7)/100.0
      FTM(13)=VPOS(8)*VRNG(8)/100.0
      UAC=VPOS(9)*VRNG(9)*(1.D0+TESUB8(9,TIME))/100.0
      FWR=VPOS(10)*VRNG(10)/100.0
      FWS=VPOS(11)*VRNG(11)/100.0
      AGSP=(VPOS(12)+150.0)/100.0
      DLP=PTV-PTR
      IF(DLP.LT.0.0)DLP=0.0
      FLMS=1937.6D0*DSQRT(DLP)
      FTM(6)=FLMS/XMWS(6)
      DLP=PTR-PTS
      IF(DLP.LT.0.0)DLP=0.0
      FLMS=4574.21D0*DSQRT(DLP)
     .*(1.D0-0.25D0*TESUB8(12,TIME))
      FTM(8)=FLMS/XMWS(8)
      DLP=PTS-760.0
      IF(DLP.LT.0.0)DLP=0.0
      FLMS=VPOS(6)*0.151169D0*DSQRT(DLP)
      FTM(10)=FLMS/XMWS(10)
      PR=PTV/PTS
      IF(PR.LT.1.0)PR=1.0
      IF(PR.GT.CPPRMX)PR=CPPRMX
      FLCOEF=CPFLMX/1.197D0
      FLMS=CPFLMX+FLCOEF*(1.0-PR**3)
      CPDH=FLMS*(TCS+273.15D0)*1.8D-6*1.9872D0
     .*(PTV-PTS)/(XMWS(9)*PTS)
      DLP=PTV-PTS
      IF(DLP.LT.0.0)DLP=0.0
      FLMS=FLMS-VPOS(5)*53.349D0*DSQRT(DLP)
      IF(FLMS.LT.1.D-3)FLMS=1.D-3
      FTM(9)=FLMS/XMWS(9)
      HST(9)=HST(9)+CPDH/FTM(9)
      DO 5020 I=1,8
      FCM(I,1)=XST(I,1)*FTM(1)
      FCM(I,2)=XST(I,2)*FTM(2)
      FCM(I,3)=XST(I,3)*FTM(3)
      FCM(I,4)=XST(I,4)*FTM(4)
      FCM(I,6)=XST(I,6)*FTM(6)
      FCM(I,8)=XST(I,8)*FTM(8)
      FCM(I,9)=XST(I,9)*FTM(9)
      FCM(I,10)=XST(I,10)*FTM(10)
      FCM(I,11)=XST(I,11)*FTM(11)
      FCM(I,13)=XST(I,13)*FTM(13)
 5020 CONTINUE
      IF(FTM(11).GT.0.1)THEN
      IF(TCC.GT.170.)THEN
      TMPFAC=TCC-120.262
      ELSEIF(TCC.LT.5.292)THEN
      TMPFAC=0.1
      ELSE
      TMPFAC=363.744/(177.-TCC)-2.22579488
      ENDIF
      VOVRL=FTM(4)/FTM(11)*TMPFAC
      SFR(4)=8.5010*VOVRL/(1.0+8.5010*VOVRL)
      SFR(5)=11.402*VOVRL/(1.0+11.402*VOVRL)
      SFR(6)=11.795*VOVRL/(1.0+11.795*VOVRL)
      SFR(7)=0.0480*VOVRL/(1.0+0.0480*VOVRL)
      SFR(8)=0.0242*VOVRL/(1.0+0.0242*VOVRL)
      ELSE
      SFR(4)=0.9999
      SFR(5)=0.999
      SFR(6)=0.999
      SFR(7)=0.99
      SFR(8)=0.98
      END IF
      DO 6010 I=1,8
      FIN(I)=0.0
      FIN(I)=FIN(I)+FCM(I,4)
      FIN(I)=FIN(I)+FCM(I,11)
 6010 CONTINUE
      FTM(5)=0.0
      FTM(12)=0.0
      DO 6020 I=1,8
      FCM(I,5)=SFR(I)*FIN(I)
      FCM(I,12)=FIN(I)-FCM(I,5)
      FTM(5)=FTM(5)+FCM(I,5)
      FTM(12)=FTM(12)+FCM(I,12)
 6020 CONTINUE
      DO 6030 I=1,8
      XST(I,5)=FCM(I,5)/FTM(5)
      XST(I,12)=FCM(I,12)/FTM(12)
 6030 CONTINUE
      TST(5)=TCC
      TST(12)=TCC
      CALL TESUB1(XST(1,5),TST(5),HST(5),1)
      CALL TESUB1(XST(1,12),TST(12),HST(12),0)
      FTM(7)=FTM(6)
      HST(7)=HST(6)
      TST(7)=TST(6)
      DO 6130 I=1,8
      XST(I,7)=XST(I,6)
      FCM(I,7)=FCM(I,6)
 6130 CONTINUE
      IF(VLR/7.8.GT.50.0)THEN
      UARLEV=1.0
      ELSEIF(VLR/7.8.LT.10.0)THEN
      UARLEV=0.0
      ELSE
      UARLEV=0.025*VLR/7.8-0.25
      ENDIF
      UAR=UARLEV*(-0.5*AGSP**2
     .+2.75*AGSP-2.5)*855490.D-6
      QUR=UAR*(TWR-TCR)
     .*(1.D0-0.35D0*TESUB8(10,TIME))
      UAS=0.404655*(1.0-1.0/(1.0+(FTM(8)/3528.73)**4))
      QUS=UAS*(TWS-TST(8))
     .*(1.D0-0.25D0*TESUB8(11,TIME))
      QUC=0.D0
      IF(TCC.LT.100.)QUC=UAC*(100.0-TCC)
      XMEAS(1)=FTM(3)*0.359/35.3145
      XMEAS(2)=FTM(1)*XMWS(1)*0.454
      XMEAS(3)=FTM(2)*XMWS(2)*0.454
      XMEAS(4)=FTM(4)*0.359/35.3145
      XMEAS(5)=FTM(9)*0.359/35.3145
      XMEAS(6)=FTM(6)*0.359/35.3145
      XMEAS(7)=(PTR-760.0)/760.0*101.325
      XMEAS(8)=(VLR-84.6)/666.7*100.0
      XMEAS(9)=TCR
      XMEAS(10)=FTM(10)*0.359/35.3145
      XMEAS(11)=TCS
      XMEAS(12)=(VLS-27.5)/290.0*100.0
      XMEAS(13)=(PTS-760.0)/760.0*101.325
      XMEAS(14)=FTM(11)/DLS/35.3145
      XMEAS(15)=(VLC-78.25)/VTC*100.0
      XMEAS(16)=(PTV-760.0)/760.0*101.325
      XMEAS(17)=FTM(13)/DLC/35.3145
      XMEAS(18)=TCC
      XMEAS(19)=QUC*1.04D3*0.454
      XMEAS(20)=CPDH*0.0003927D6
      XMEAS(20)=CPDH*0.29307D3
      XMEAS(21)=TWR
      XMEAS(22)=TWS
      ISD=0
      IF(XMEAS(7).GT.3000.0)ISD=1
      IF(VLR/35.3145.GT.24.0)ISD=1
      IF(VLR/35.3145.LT.2.0)ISD=1
      IF(XMEAS(9).GT.175.0)ISD=1
      IF(VLS/35.3145.GT.12.0)ISD=1
      IF(VLS/35.3145.LT.1.0)ISD=1
      IF(VLC/35.3145.GT.8.0)ISD=1
      IF(VLC/35.3145.LT.1.0)ISD=1
      IF(TIME.GT.0.0.AND.ISD.EQ.0)THEN
      DO 6500 I=1,22
      CALL TESUB6(XNS(I),XMNS)
      XMEAS(I)=XMEAS(I)+XMNS
 6500 CONTINUE
      ENDIF
      XCMP(23)=XST(1,7)*100.0
      XCMP(24)=XST(2,7)*100.0
      XCMP(25)=XST(3,7)*100.0
      XCMP(26)=XST(4,7)*100.0
      XCMP(27)=XST(5,7)*100.0
      XCMP(28)=XST(6,7)*100.0
      XCMP(29)=XST(1,10)*100.0
      XCMP(30)=XST(2,10)*100.0
      XCMP(31)=XST(3,10)*100.0
      XCMP(32)=XST(4,10)*100.0
      XCMP(33)=XST(5,10)*100.0
      XCMP(34)=XST(6,10)*100.0
      XCMP(35)=XST(7,10)*100.0
      XCMP(36)=XST(8,10)*100.0
      XCMP(37)=XST(4,13)*100.0
      XCMP(38)=XST(5,13)*100.0
      XCMP(39)=XST(6,13)*100.0
      XCMP(40)=XST(7,13)*100.0
      XCMP(41)=XST(8,13)*100.0
      IF(TIME.EQ.0.D0)THEN
      DO 7010 I=23,41
      XDEL(I)=XCMP(I)
      XMEAS(I)=XCMP(I)
 7010 CONTINUE
      TGAS=0.1
      TPROD=0.25
      ENDIF
      IF(TIME.GE.TGAS)THEN
      DO 7020 I=23,36
      XMEAS(I)=XDEL(I)
      CALL TESUB6(XNS(I),XMNS)
      XMEAS(I)=XMEAS(I)+XMNS
      XDEL(I)=XCMP(I)
 7020 CONTINUE
      TGAS=TGAS+0.1
      ENDIF
      IF(TIME.GE.TPROD)THEN
      DO 7030 I=37,41
      XMEAS(I)=XDEL(I)
      CALL TESUB6(XNS(I),XMNS)
      XMEAS(I)=XMEAS(I)+XMNS
      XDEL(I)=XCMP(I)
 7030 CONTINUE
      TPROD=TPROD+0.25
      ENDIF
      DO 9010 I=1,8
      YP(I)=FCM(I,7)-FCM(I,8)+CRXR(I)
      YP(I+9)=FCM(I,8)-FCM(I,9)-
     .FCM(I,10)-FCM(I,11)
      YP(I+18)=FCM(I,12)-FCM(I,13)
      YP(I+27)=FCM(I,1)+FCM(I,2)+
     .FCM(I,3)+FCM(I,5)+
     .FCM(I,9)-FCM(I,6)
 9010 CONTINUE
      YP(9)=HST(7)*FTM(7)-
     .HST(8)*FTM(8)+RH+QUR
      YP(18)=HST(8)*FTM(8)-
     .HST(9)*FTM(9)-
     .HST(10)*FTM(10)-
     .HST(11)*FTM(11)+
     .QUS
      YP(27)=HST(4)*FTM(4)+
     .HST(11)*FTM(11)-
     .HST(5)*FTM(5)-
     .HST(13)*FTM(13)+
     .QUC
      YP(36)=HST(1)*FTM(1)+
     .HST(2)*FTM(2)+
     .HST(3)*FTM(3)+
     .HST(5)*FTM(5)+
     .HST(9)*FTM(9)-
     .HST(6)*FTM(6)
      YP(37)=(FWR*500.53*
     .(TCWR-TWR)-QUR*1.D6/1.8)/HWR
      YP(38)=(FWS*500.53*
     .(TCWS-TWS)-QUS*1.D6/1.8)/HWS
      IVST(10)=IDV(14)
      IVST(11)=IDV(15)
      IVST(5)=IDV(19)
      IVST(7)=IDV(19)
      IVST(8)=IDV(19)
      IVST(9)=IDV(19)
      DO 9020 I=1,12
      IF(TIME.EQ.0.D0 .OR.
     .DABS(VCV(I)-XMV(I)).GT.VST(I)*IVST(I))
     .VCV(I)=XMV(I)
      IF(VCV(I).LT.0.0)VCV(I)=0.0
      IF(VCV(I).GT.100.0)VCV(I)=100.0
      YP(I+38)=(VCV(I)-VPOS(I))/VTAU(I)
 9020 CONTINUE
      IF(ISD.NE.0)THEN
      DO 9030 I=1,NN
      YP(I)=0.0
 9030 CONTINUE
      ENDIF
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE TEINIT(NN,TIME,YY,YP,RANDOMSEED)
      IMPLICIT NONE
C
C       Initialization
C
C         Inputs:
C
C           NN   = Number of differential equations
C
C         Outputs:
C
C           Time = Current time(hrs)
C           YY   = Current state values
C           YP   = Current derivative values
C           RANDOMSEED = Seed for random generator, zero means use random generator
C
      DOUBLE PRECISION RANDOMSEED
      DOUBLE PRECISION XMEAS,XMV
      COMMON/PV/XMEAS(41),XMV(12)
      INTEGER IDV
      COMMON/DVEC/IDV(20)
      DOUBLE PRECISION G
      COMMON/RANDSD/G
      DOUBLE PRECISION
     .UCLR,UCVR,UTLR,UTVR,
     .XLR,XVR,ETR,ESR,
     .TCR,TKR,DLR,
     .VLR,VVR,VTR,
     .PTR,PPR,
     .CRXR,RR,RH,
     .FWR,TWR,QUR,HWR,UAR,
     .UCLS,UCVS,UTLS,UTVS,
     .XLS,XVS,ETS,ESS,
     .TCS,TKS,DLS,
     .VLS,VVS,VTS,
     .PTS,PPS,
     .FWS,TWS,QUS,HWS,
     .UCLC,UTLC,XLC,
     .ETC,ESC,TCC,DLC,
     .VLC,VTC,QUC,
     .UCVV,UTVV,XVV,
     .ETV,ESV,TCV,TKV,
     .VTV,PTV,
     .VCV,VRNG,VTAU,
     .FTM,
     .FCM,XST,XMWS,
     .HST,TST,SFR,
     .CPFLMX,CPPRMX,CPDH,
     .TCWR,TCWS,
     .HTR,AGSP,
     .XDEL,XNS,
     .TGAS,TPROD,VST
      INTEGER
     .IVST
      COMMON/TEPROC/
     .UCLR(8),UCVR(8),UTLR,UTVR,
     .XLR(8),XVR(8),ETR,ESR,
     .TCR,TKR,DLR,
     .VLR,VVR,VTR,
     .PTR,PPR(8),
     .CRXR(8),RR(4),RH,
     .FWR,TWR,QUR,HWR,UAR,
     .UCLS(8),UCVS(8),UTLS,UTVS,
     .XLS(8),XVS(8),ETS,ESS,
     .TCS,TKS,DLS,
     .VLS,VVS,VTS,
     .PTS,PPS(8),
     .FWS,TWS,QUS,HWS,
     .UCLC(8),UTLC,XLC(8),
     .ETC,ESC,TCC,DLC,
     .VLC,VTC,QUC,
     .UCVV(8),UTVV,XVV(8),
     .ETV,ESV,TCV,TKV,
     .VTV,PTV,
     .VCV(12),VRNG(12),VTAU(12),
     .FTM(13),
     .FCM(8,13),XST(8,13),XMWS(13),
     .HST(13),TST(13),SFR(8),
     .CPFLMX,CPPRMX,CPDH,
     .TCWR,TCWS,
     .HTR(3),AGSP,
     .XDEL(41),XNS(41),
     .TGAS,TPROD,VST(12),IVST(12)
      INTEGER IDVWLK
      DOUBLE PRECISION
     .ADIST,
     .BDIST,
     .CDIST,
     .DDIST,
     .TLAST,
     .TNEXT,
     .HSPAN,
     .HZERO,
     .SSPAN,
     .SZERO,
     .SPSPAN
      COMMON/WLK/
     .ADIST(12),
     .BDIST(12),
     .CDIST(12),
     .DDIST(12),
     .TLAST(12),
     .TNEXT(12),
     .HSPAN(12),
     .HZERO(12),
     .SSPAN(12),
     .SZERO(12),
     .SPSPAN(12),
     .IDVWLK(12)
      DOUBLE PRECISION
     .AVP,BVP,CVP,
     .AH,BH,CH,
     .AG,BG,CG,
     .AV,
     .AD,BD,CD,
     .XMW
      COMMON/CONST/
     .AVP(8),BVP(8),CVP(8),
     .AH(8),BH(8),CH(8),
     .AG(8),BG(8),CG(8),
     .AV(8),
     .AD(8),BD(8),CD(8),
     .XMW(8)
      INTEGER I,NN
      DOUBLE PRECISION YY(NN),
     .YP(NN),
     .TIME

C      write(*,*) 'TEINIT> Init...'

      XMW(1)=2.0
      XMW(2)=25.4
      XMW(3)=28.0
      XMW(4)=32.0
      XMW(5)=46.0
      XMW(6)=48.0
      XMW(7)=62.0
      XMW(8)=76.0
      AVP(1)=0.0
      AVP(2)=0.0
      AVP(3)=0.0
      AVP(4)=15.92
      AVP(5)=16.35
      AVP(6)=16.35
      AVP(7)=16.43
      AVP(8)=17.21
      BVP(1)=0.0
      BVP(2)=0.0
      BVP(3)=0.0
      BVP(4)=-1444.0
      BVP(5)=-2114.0
      BVP(6)=-2114.0
      BVP(7)=-2748.0
      BVP(8)=-3318.0
      CVP(1)=0.0
      CVP(2)=0.0
      CVP(3)=0.0
      CVP(4)=259.0
      CVP(5)=265.5
      CVP(6)=265.5
      CVP(7)=232.9
      CVP(8)=249.6
      AD(1)=1.0
      AD(2)=1.0
      AD(3)=1.0
      AD(4)=23.3
      AD(5)=33.9
      AD(6)=32.8
      AD(7)=49.9
      AD(8)=50.5
      BD(1)=0.0
      BD(2)=0.0
      BD(3)=0.0
      BD(4)=-0.0700
      BD(5)=-0.0957
      BD(6)=-0.0995
      BD(7)=-0.0191
      BD(8)=-0.0541
      CD(1)=0.0
      CD(2)=0.0
      CD(3)=0.0
      CD(4)=-0.0002
      CD(5)=-0.000152
      CD(6)=-0.000233
      CD(7)=-0.000425
      CD(8)=-0.000150
      AH(1)=1.0D-6
      AH(2)=1.0D-6
      AH(3)=1.0D-6
      AH(4)=0.960D-6
      AH(5)=0.573D-6
      AH(6)=0.652D-6
      AH(7)=0.515D-6
      AH(8)=0.471D-6
      BH(1)=0.0
      BH(2)=0.0
      BH(3)=0.0
      BH(4)=8.70D-9
      BH(5)=2.41D-9
      BH(6)=2.18D-9
      BH(7)=5.65D-10
      BH(8)=8.70D-10
      CH(1)=0.0
      CH(2)=0.0
      CH(3)=0.0
      CH(4)=4.81D-11
      CH(5)=1.82D-11
      CH(6)=1.94D-11
      CH(7)=3.82D-12
      CH(8)=2.62D-12
      AV(1)=1.0D-6
      AV(2)=1.0D-6
      AV(3)=1.0D-6
      AV(4)=86.7D-6
      AV(5)=160.D-6
      AV(6)=160.D-6
      AV(7)=225.D-6
      AV(8)=209.D-6
      AG(1)=3.411D-6
      AG(2)=0.3799D-6
      AG(3)=0.2491D-6
      AG(4)=0.3567D-6
      AG(5)=0.3463D-6
      AG(6)=0.3930D-6
      AG(7)=0.170D-6
      AG(8)=0.150D-6
      BG(1)=7.18D-10
      BG(2)=1.08D-9
      BG(3)=1.36D-11
      BG(4)=8.51D-10
      BG(5)=8.96D-10
      BG(6)=1.02D-9
      BG(7)=0.D0
      BG(8)=0.D0
      CG(1)=6.0D-13
      CG(2)=-3.98D-13
      CG(3)=-3.93D-14
      CG(4)=-3.12D-13
      CG(5)=-3.27D-13
      CG(6)=-3.12D-13
      CG(7)=0.D0
      CG(8)=0.D0
      YY(1)=10.40491389
      YY(2)=4.363996017
      YY(3)=7.570059737
      YY(4)=0.4230042431
      YY(5)=24.15513437
      YY(6)=2.942597645
      YY(7)=154.3770655
      YY(8)=159.1865960
      YY(9)=2.808522723
      YY(10)=63.75581199
      YY(11)=26.74026066
      YY(12)=46.38532432
      YY(13)=0.2464521543
      YY(14)=15.20484404
      YY(15)=1.852266172
      YY(16)=52.44639459
      YY(17)=41.20394008
      YY(18)=0.5699317760
      YY(19)=0.4306056376
      YY(20)=7.9906200783D-03
      YY(21)=0.9056036089
      YY(22)=1.6054258216D-02
      YY(23)=0.7509759687
      YY(24)=8.8582855955D-02
      YY(25)=48.27726193
      YY(26)=39.38459028
      YY(27)=0.3755297257
      YY(28)=107.7562698
      YY(29)=29.77250546
      YY(30)=88.32481135
      YY(31)=23.03929507
      YY(32)=62.85848794
      YY(33)=5.546318688
      YY(34)=11.92244772
      YY(35)=5.555448243
      YY(36)=0.9218489762
      YY(37)=94.59927549
      YY(38)=77.29698353
      YY(39)=63.05263039
      YY(40)=53.97970677
      YY(41)=24.64355755
      YY(42)=61.30192144
      YY(43)=22.21000000
      YY(44)=40.06374673
      YY(45)=38.10034370
      YY(46)=46.53415582
      YY(47)=47.44573456
      YY(48)=41.10581288
      YY(49)=18.11349055
      YY(50)=50.00000000
      DO 200 I=1,12
      XMV(I)=YY(I+38)
      VCV(I)=XMV(I)
      VST(I)=2.0D0
      IVST(I)=0
 200  CONTINUE
      VRNG(1)=400.00
      VRNG(2)=400.00
      VRNG(3)=100.00
      VRNG(4)=1500.00
      VRNG(7)=1500.00
      VRNG(8)=1000.00
      VRNG(9)=0.03
      VRNG(10)=1000.
      VRNG(11)=1200.0
      VTR=1300.0
      VTS=3500.0
      VTC=156.5
      VTV=5000.0
      HTR(1)=0.06899381054D0
      HTR(2)=0.05D0
      HWR=7060.
      HWS=11138.
      SFR(1)=0.99500
      SFR(2)=0.99100
      SFR(3)=0.99000
      SFR(4)=0.91600
      SFR(5)=0.93600
      SFR(6)=0.93800
      SFR(7)=5.80000D-02
      SFR(8)=3.01000D-02
      XST(1,1)=0.0
      XST(2,1)=0.0001
      XST(3,1)=0.0
      XST(4,1)=0.9999
      XST(5,1)=0.0
      XST(6,1)=0.0
      XST(7,1)=0.0
      XST(8,1)=0.0
      TST(1)=45.
      XST(1,2)=0.0
      XST(2,2)=0.0
      XST(3,2)=0.0
      XST(4,2)=0.0
      XST(5,2)=0.9999
      XST(6,2)=0.0001
      XST(7,2)=0.0
      XST(8,2)=0.0
      TST(2)=45.
      XST(1,3)=0.9999
      XST(2,3)=0.0001
      XST(3,3)=0.0
      XST(4,3)=0.0
      XST(5,3)=0.0
      XST(6,3)=0.0
      XST(7,3)=0.0
      XST(8,3)=0.0
      TST(3)=45.
      XST(1,4)=0.4850
      XST(2,4)=0.0050
      XST(3,4)=0.5100
      XST(4,4)=0.0
      XST(5,4)=0.0
      XST(6,4)=0.0
      XST(7,4)=0.0
      XST(8,4)=0.0
      TST(4)=45.
      CPFLMX=280275.
      CPPRMX=1.3
      VTAU(1)=8.
      VTAU(2)=8.
      VTAU(3)=6.
      VTAU(4)=9.
      VTAU(5)=7.
      VTAU(6)=5.
      VTAU(7)=5.
      VTAU(8)=5.
      VTAU(9)=120.
      VTAU(10)=5.
      VTAU(11)=5.
      VTAU(12)=5.
      DO 300 I=1,12
      VTAU(I)=VTAU(I)/3600.
 300  CONTINUE
C      write(*,*) 'Random number generation: seed=',RANDOMSEED
      G=RANDOMSEED
C     G=4651207995.D0                 ! ORIGINAL
C	d00_tr_new: G=5687912315.D0                        ! ORIGINAL
C      original: G=1431655765.D0                 ! ORIGINAL
C        d00_tr: G=4243534565.D0                 ! ORIGINAL
C        d01_tr: G=7854912354.D0                 ! ORIGINAL
C        d02_tr: G=3456432354.D0                 ! ORIGINAL
C        d03_tr: G=1731738903.D0                 ! ORIGINAL
C        d04_tr: G=4346024432.D0                 ! ORIGINAL
C        d05_tr: G=5784921734.D0                 ! ORIGINAL
C        d06_tr: G=6678322168.D0                 ! ORIGINAL
C        d07_tr: G=7984782901.D0                 ! ORIGINAL
C        d08_tr: G=8934302332.D0                 ! ORIGINAL
C        d09_tr: G=9873223412.D0                 ! ORIGINAL
C        d10_tr: G=1089278833.D0                 ! ORIGINAL
C        d11_tr: G=1940284333.D0                 ! ORIGINAL
C        d12_tr: G=2589274931.D0                 ! ORIGINAL
C        d13_tr: G=3485834345.D0                 ! ORIGINAL
C        d14_tr: G=4593493842.D0                 ! ORIGINAL
C        d15_tr: G=5683213434.D0                 ! ORIGINAL
C        d16_tr: G=6788343442.D0                 ! ORIGINAL
C        d17_tr: G=1723234455.D0                 ! ORIGINAL
C        d18_tr: G=8943243993.D0                 ! ORIGINAL
C       dd18_tr: G=1234567890.D0                 ! ORIGINAL

C        d19_tr: G=9445382439.D0                 ! ORIGINAL
C        d20_tr: G=9902234324.D0                 ! ORIGINAL
C        d21_tr: G=2144342545.D0                 ! ORIGINAL
C        d22_tr: G=3433249064.D0                 ! ORIGINAL
C        d23_tr: G=4356565463.D0                 ! ORIGINAL
C        d24_tr: G=8998485332.D0                 ! ORIGINAL
C        d25_tr: G=7654534567.D0                 ! ORIGINAL
C        d26_tr: G=5457789234.D0                 ! ORIGINAL
C
C        d00_te: G=1254545354.D0                 ! ORIGINAL
C        d01_te: G=2994833239.D0                 ! ORIGINAL
C        d02_te: G=2891123453.D0                 ! ORIGINAL
C        d03_te: G=3420494299.D0                 ! ORIGINAL
C        d04_te: G=4598956239.D0                 ! ORIGINAL
C        d05_te: G=5658678765.D0                 ! ORIGINAL
C        d06_te: G=6598593453.D0                 ! ORIGINAL
C        d07_te: G=7327843434.D0                 ! ORIGINAL
C        d08_te: G=8943242344.D0                 ! ORIGINAL
C        d09_te: G=9343430004.D0                 ! ORIGINAL
C        d10_te: G=1039839281.D0                 ! ORIGINAL
C        d11_te: G=1134345551.D0                 ! ORIGINAL
C        d12_te: G=2232323236.D0                 ! ORIGINAL
C        d13_te: G=3454354353.D0                 ! ORIGINAL
C        d14_te: G=4545445883.D0                 ! ORIGINAL
C        d15_te: G=5849489384.D0                 ! ORIGINAL
C        d16_te: G=6284545932.D0                 ! ORIGINAL
C        d17_te: G=4342232344.D0                 ! ORIGINAL
C        d18_te: G=5635346588.D0                 ! ORIGINAL
C        d19_te: G=9090909232.DO                 ! ORIGINAL
C        d20_te: G=8322308324.D0                 ! ORIGINAL
C        d21_te: G=2132432423.D0                 ! ORIGINAL
C        d22_te: G=5454589923.D0                 ! ORIGINAL
C        d23_te: G=6923255678.D0                 ! ORIGINAL
C        d24_te: G=8493323434.D0                 ! ORIGINAL
C        d25_te: G=9338398429.D0                 ! ORIGINAL
C        d26_te: G=1997072199.D0                 ! ORIGINAL

      XNS(1)=0.0012D0
      XNS(2)=18.000D0
      XNS(3)=22.000D0
      XNS(4)=0.0500D0
      XNS(5)=0.2000D0
      XNS(6)=0.2100D0
      XNS(7)=0.3000D0
      XNS(8)=0.5000D0
      XNS(9)=0.0100D0
      XNS(10)=0.0017D0
      XNS(11)=0.0100D0
      XNS(12)=1.0000D0
      XNS(13)=0.3000D0
      XNS(14)=0.1250D0
      XNS(15)=1.0000D0
      XNS(16)=0.3000D0
      XNS(17)=0.1150D0
      XNS(18)=0.0100D0
      XNS(19)=1.1500D0
      XNS(20)=0.2000D0
      XNS(21)=0.0100D0
      XNS(22)=0.0100D0
      XNS(23)=0.250D0
      XNS(24)=0.100D0
      XNS(25)=0.250D0
      XNS(26)=0.100D0
      XNS(27)=0.250D0
      XNS(28)=0.025D0
      XNS(29)=0.250D0
      XNS(30)=0.100D0
      XNS(31)=0.250D0
      XNS(32)=0.100D0
      XNS(33)=0.250D0
      XNS(34)=0.025D0
      XNS(35)=0.050D0
      XNS(36)=0.050D0
      XNS(37)=0.010D0
      XNS(38)=0.010D0
      XNS(39)=0.010D0
      XNS(40)=0.500D0
      XNS(41)=0.500D0
      DO 500 I=1,20
      IDV(I)=0
 500  CONTINUE
      HSPAN(1)=0.2D0
      HZERO(1)=0.5D0
      SSPAN(1)=0.03D0
      SZERO(1)=0.485D0
      SPSPAN(1)=0.D0
      HSPAN(2)=0.7D0
      HZERO(2)=1.0D0
      SSPAN(2)=.003D0
      SZERO(2)=.005D0
      SPSPAN(2)=0.D0
      HSPAN(3)=0.25D0
      HZERO(3)=0.5D0
      SSPAN(3)=10.D0
      SZERO(3)=45.D0
      SPSPAN(3)=0.D0
      HSPAN(4)=0.7D0
      HZERO(4)=1.0D0
      SSPAN(4)=10.D0
      SZERO(4)=45.D0
      SPSPAN(4)=0.D0
      HSPAN(5)=0.15D0
      HZERO(5)=0.25D0
      SSPAN(5)=10.D0
      SZERO(5)=35.D0
      SPSPAN(5)=0.D0
      HSPAN(6)=0.15D0
      HZERO(6)=0.25D0
      SSPAN(6)=10.D0
      SZERO(6)=40.D0
      SPSPAN(6)=0.D0
      HSPAN(7)=1.D0
      HZERO(7)=2.D0
      SSPAN(7)=0.25D0
      SZERO(7)=1.0D0
      SPSPAN(7)=0.D0
      HSPAN(8)=1.D0
      HZERO(8)=2.D0
      SSPAN(8)=0.25D0
      SZERO(8)=1.0D0
      SPSPAN(8)=0.D0
      HSPAN(9)=0.4D0
      HZERO(9)=0.5D0
      SSPAN(9)=0.25D0
      SZERO(9)=0.0D0
      SPSPAN(9)=0.D0
      HSPAN(10)=1.5D0
      HZERO(10)=2.0D0
      SSPAN(10)=0.0D0
      SZERO(10)=0.0D0
      SPSPAN(10)=0.D0
      HSPAN(11)=2.0D0
      HZERO(11)=3.0D0
      SSPAN(11)=0.0D0
      SZERO(11)=0.0D0
      SPSPAN(11)=0.D0
      HSPAN(12)=1.5D0
      HZERO(12)=2.0D0
      SSPAN(12)=0.0D0
      SZERO(12)=0.0D0
      SPSPAN(12)=0.D0
      DO 550 I=1,12
      TLAST(I)=0.D0
      TNEXT(I)=0.1D0
      ADIST(I)=SZERO(I)
      BDIST(I)=0.D0
      CDIST(I)=0.D0
      DDIST(I)=0.D0
  550 CONTINUE
      TIME=0.0
      CALL TEFUNC(NN,TIME,YY,YP)
      RETURN
      END
C
C=============================================================================
C
      SUBROUTINE TESUB1(Z,T,H,ITY)
      IMPLICIT NONE
      DOUBLE PRECISION
     .AVP,BVP,CVP,
     .AH,BH,CH,
     .AG,BG,CG,
     .AV,
     .AD,BD,CD,
     .XMW
      COMMON/CONST/
     .AVP(8),BVP(8),CVP(8),
     .AH(8),BH(8),CH(8),
     .AG(8),BG(8),CG(8),
     .AV(8),
     .AD(8),BD(8),CD(8),
     .XMW(8)
      DOUBLE PRECISION Z(8),R,T,H,HI
      INTEGER ITY,I
      IF(ITY.EQ.0)THEN
      H=0.0D0
      DO 100 I=1,8
      HI=T*(AH(I)+BH(I)*T/2.D0+CH(I)*T**2/3.D0)
      HI=1.8D0*HI
      H=H+Z(I)*XMW(I)*HI
 100  CONTINUE
      ELSE
      H=0.0D0
      DO 200 I=1,8
      HI=T*(AG(I)+BG(I)*T/2.D0+
     .CG(I)*T**2/3.D0)
      HI=1.8D0*HI
      HI=HI+AV(I)
      H=H+Z(I)*XMW(I)*HI
 200  CONTINUE
      END IF
      IF(ITY.EQ.2)THEN
      R=3.57696D0/1.D6
      H=H-R*(T+273.15)
      ENDIF
      RETURN
      END

      SUBROUTINE TESUB2(Z,T,H,ITY)
      IMPLICIT NONE
      DOUBLE PRECISION
     .AVP,BVP,CVP,
     .AH,BH,CH,
     .AG,BG,CG,
     .AV,
     .AD,BD,CD,
     .XMW
      COMMON/CONST/
     .AVP(8),BVP(8),CVP(8),
     .AH(8),BH(8),CH(8),
     .AG(8),BG(8),CG(8),
     .AV(8),
     .AD(8),BD(8),CD(8),
     .XMW(8)
      INTEGER ITY,J
      DOUBLE PRECISION Z(8),T,H,TIN,HTEST,ERR,DH,DT
      TIN=T
      DO 250 J=1,100
      CALL TESUB1(Z,T,HTEST,ITY)
      ERR=HTEST-H
      CALL TESUB3(Z,T,DH,ITY)
      DT=-ERR/DH
      T=T+DT
 250  IF(DABS(DT).LT.1.D-12)GO TO 300
      T=TIN
 300  RETURN
      END

      SUBROUTINE TESUB3(Z,T,DH,ITY)
      IMPLICIT NONE
      DOUBLE PRECISION
     .AVP,BVP,CVP,
     .AH,BH,CH,
     .AG,BG,CG,
     .AV,
     .AD,BD,CD,
     .XMW
      COMMON/CONST/
     .AVP(8),BVP(8),CVP(8),
     .AH(8),BH(8),CH(8),
     .AG(8),BG(8),CG(8),
     .AV(8),
     .AD(8),BD(8),CD(8),
     .XMW(8)
      INTEGER ITY,I
      DOUBLE PRECISION Z(8),R,T,DH,DHI
      IF(ITY.EQ.0)THEN
      DH=0.0D0
      DO 100 I=1,8
      DHI=AH(I)+BH(I)*T+CH(I)*T**2
      DHI=1.8D0*DHI
      DH=DH+Z(I)*XMW(I)*DHI
 100  CONTINUE
      ELSE
      DH=0.0D0
      DO 200 I=1,8
      DHI=AG(I)+BG(I)*T+CG(I)*T**2
      DHI=1.8D0*DHI
      DH=DH+Z(I)*XMW(I)*DHI
 200  CONTINUE
      END IF
      IF(ITY.EQ.2)THEN
      R=3.57696D0/1.D6
      DH=DH-R
      ENDIF
      RETURN
      END
      
      SUBROUTINE TESUB4(X,T,R)
      IMPLICIT NONE
      DOUBLE PRECISION
     .AVP,BVP,CVP,
     .AH,BH,CH,
     .AG,BG,CG,
     .AV,
     .AD,BD,CD,
     .XMW
      COMMON/CONST/
     .AVP(8),BVP(8),CVP(8),
     .AH(8),BH(8),CH(8),
     .AG(8),BG(8),CG(8),
     .AV(8),
     .AD(8),BD(8),CD(8),
     .XMW(8)
      DOUBLE PRECISION V,R,X(8),T
      INTEGER I
      V=0.0
      DO 10 I=1,8
      V=V+X(I)*XMW(I)/
     .(AD(I)+(BD(I)+CD(I)*T)*T)
   10 CONTINUE
      R=1.0/V
      RETURN
      END

      SUBROUTINE TESUB5(S,SP,ADIST,BDIST,CDIST,DDIST,TLAST,
     .TNEXT,HSPAN,HZERO,SSPAN,SZERO,SPSPAN,IDVFLAG)
      IMPLICIT NONE
      DOUBLE PRECISION
     .S,
     .SP,
     .H,
     .S1,
     .S1P,
     .ADIST,
     .BDIST,
     .CDIST,
     .DDIST,
     .TLAST,
     .TNEXT,
     .HSPAN,
     .HZERO,
     .SSPAN,
     .SZERO,
     .SPSPAN,
     .TESUB7
      INTEGER  I,IDVFLAG
      I=-1
      H=HSPAN*TESUB7(I)+HZERO
      S1=SSPAN*TESUB7(I)*IDVFLAG+SZERO
      S1P=SPSPAN*TESUB7(I)*IDVFLAG
      ADIST=S
      BDIST=SP
      CDIST=(3.D0*(S1-S)-H*(S1P+2.D0*SP))/H**2
      DDIST=(2.D0*(S-S1)+H*(S1P+SP))/H**3
      TNEXT=TLAST+H
      RETURN
      END

      SUBROUTINE TESUB6(STD,X)
      IMPLICIT NONE
      INTEGER I
      DOUBLE PRECISION STD,X,TESUB7
      X=0.D0
      DO 2 I=1,12
    2 X=X+TESUB7(I)
      X=(X-6.D0)*STD
      RETURN
      END

      DOUBLE PRECISION FUNCTION TESUB7(I)
      IMPLICIT NONE
      INTEGER I
      DOUBLE PRECISION G,DMOD
      COMMON/RANDSD/G
      G=DMOD(G*9228907.D0,4294967296.D0)
      IF(I.GE.0)TESUB7=G/4294967296.D0
      IF(I.LT.0)TESUB7=2.D0*G/4294967296.D0-1.D0
      RETURN
      END
      DOUBLE PRECISION FUNCTION TESUB8(I,T)
      INTEGER  I
      DOUBLE PRECISION  H,T
      INTEGER IDVWLK
      DOUBLE PRECISION
     .ADIST,
     .BDIST,
     .CDIST,
     .DDIST,
     .TLAST,
     .TNEXT,
     .HSPAN,
     .HZERO,
     .SSPAN,
     .SZERO,
     .SPSPAN
      COMMON/WLK/
     .ADIST(12),
     .BDIST(12),
     .CDIST(12),
     .DDIST(12),
     .TLAST(12),
     .TNEXT(12),
     .HSPAN(12),
     .HZERO(12),
     .SSPAN(12),
     .SZERO(12),
     .SPSPAN(12),
     .IDVWLK(12)
      H=T-TLAST(I)
      TESUB8=ADIST(I)+H*(BDIST(I)+H*(CDIST(I)+H*DDIST(I)))
      RETURN
      END

C
      PROGRAM TEGUESSSEED
      IMPLICIT NONE
C       Show number of features in CSV data file
C     In configuration file, 0=Generate randomly
      DOUBLE PRECISION RANDOMSEED
C
C  MEASUREMENT AND VALVE COMMON BLOCK
C
      DOUBLE PRECISION XMEAS, XMV, xmeascompare(41)
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
      INTEGER I, J, NN, NPTS, TEST, TEST1, TEST4
      double precision RS, maxrs
C
      DOUBLE PRECISION TIME, YY(50), YP(50)
      CHARACTER *80 datafilename
C
C
      integer nsmp    ! Number of sampled points

      NN = 50
      DELTAT = 1. / 3600.
C=============================================================================
C
C  Set all Disturbance Flags to OFF
C
      DO 100 I = 1, 20
          IDV(I) = 0
 100  CONTINUE
C      write(*,"(1X,A,A1,20(I1))"),"INIT: IDV=","C",(IDV(J),J=1,20)
C

C>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C  Set the number of pints to simulate in steady state operation
      RANDOMSEED = 4243534565D0
      datafilename = '../../TE_process/data/d00.dat' ! Verified

      RANDOMSEED = 2994833239.D0
      datafilename = '../../TE_process/data/d01_te.dat' ! Verified

      RANDOMSEED = 2891123453D0
      datafilename = '../../TE_process/data/d02_te.dat' ! Verified

      RANDOMSEED = 3420494299.D0
      datafilename = '../../TE_process/data/d03_te.dat' ! Verified


      RANDOMSEED = 2891123453D0
      datafilename = '../../TE_process/data/d00.dat'

      RANDOMSEED = 7854912354.D0
      datafilename = '../../TE_process/data/d01.dat'

      RANDOMSEED = 3456432354.D0
      datafilename = '../../TE_process/data/d02.dat'

      RANDOMSEED = 1731738903.D0
      datafilename = '../../TE_process/data/d03.dat'

      open(unit=9,FILE=datafilename,STATUS='old')
C      write(*,*) 'Reading data file ''',datafilename,''''
      read(9,*) (xmeascompare(i),i=1,41)
C      write(*,"(1X,A,41(1P,E13.5))"),"dat=",(xmeascompare(i),i=1,41)
C Read first sample
      close(unit=9)
      maxrs = 10000000000D0
      rs =     1000000000D0

      DO WHILE (rs .lt. maxrs)
      rs = rs + 1D0
      RANDOMSEED = RS
C  Set the number of points to simulate
      NPTS = 180 ! Must be multiples of 180
C  Initialize Process
      CALL TEINIT(NN,TIME,YY,YP,RANDOMSEED)
C  Eventually set fault at the beginning: Has to be done after TEINIT
      IDV(3) = 1
C      write(*,"(1X,A,A1,20(I1))"),"SET FAULT: IDV=","C",(IDV(J),J=1,20)
      nsmp = 0
C<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
C
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

C
C
C  Simulation Loop
C
C      write(*,*) 'Starting simulation...'

      DO 1000 I = 1, NPTS
      TEST4=MOD(I,180)  ! write each 180 steps
      IF (TEST4.EQ.0) THEN
C        write(0,"(1x,A16,I8,A4,I8)") 'Simulation step ', i, ' of ', NPTS
      ENDIF

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
C
      IF (TEST4.EQ.0) THEN
C      IF (.true.) THEN
        CALL VERIFYRANDOMSEED(XMEAS,xmeascompare,
     +                        RANDOMSEED,datafilename)
            nsmp = nsmp + 1   ! Increment number of sampled points
C            write(*,"(1X,A,I8,A,I8,A,A1,20(I1))"), "Sampling point #",
C     +            nsmp, " at t=", I, " sec  Class=","C",(IDV(J),J=1,20)
      ENDIF

C
C
      CALL INTGTR(NN,TIME,DELTAT,YY,YP)
C
      CALL CONSHAND
C
 1000 CONTINUE
C      PRINT *, ''
C      PRINT *, 'Simulation is done. '

C
      ENDDO ! For all random seeds
      END ! Programm
C==========================================================================
C==========================================================================
C==========================================================================
C
      SUBROUTINE VERIFYRANDOMSEED(XMEAS,xmeascompare,
     +                            RANDOMSEED,datafilename)
      IMPLICIT NONE
      DOUBLE PRECISION XMEAS(41), xmeascompare(41), RANDOMSEED
      character*80 datafilename
      DOUBLE PRECISION res, dif, tol
      INTEGER i
C
      tol = 1e-1
      res = 0D0
C
      do i = 1, 41
         dif = XMEAS(i)-xmeascompare(i)
C         write(*,"(1x,A,I2,A,E13.5,A,E13.5,A,E13.5)")
C     +      'dif(',i,') =',XMEAS(i), ' -', xmeascompare(i), '=', dif
         res = res + dif*dif
      enddo
      res = sqrt(res)
      if ( MOD(RANDOMSEED,10000D0).eq.0) then
        write(*,*) '>>> Residual = ', res, ' Seed=', RANDOMSEED
      endif
      if (res .le. tol) then
             write(*,*) '--------------> YEPYEPYEP SEED=', RANDOMSEED,
     +              'File=', datafilename
             stop
      endif
      END
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
C            write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  6 CHANGE:',
C     +                SETPT(6),'-->',0.33712
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
C            write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  6 CHANGE:',
C     +                SETPT(6),'-->',0.33712
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
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  3 CHANGE:',
C     +                SETPT(3),'-->',SETPT(3) + DXMV * 1.017 / 100.
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
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  1 CHANGE:',
C     +                SETPT(1),'-->',SETPT(1) + DXMV * 5811. / 100.
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
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  2 CHANGE:',
C     +                SETPT(2),'-->',SETPT(2) + DXMV * 8354. / 100.
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
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  9 CHANGE:',
C     +                SETPT(9),'-->',SETPT(9) + DXMV * 460. / 100.
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
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  4 CHANGE:',
C     +                SETPT(4),'-->',SETPT(4) + DXMV * 15.25 / 100.
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
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT 10 CHANGE:',
C     +                SETPT(10),'-->',SETPT(10) + DXMV * 150. / 100.
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
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT  6 CHANGE:',
C     +                SETPT(6),'-->', SETPT(6) + DXMV * 1. / 100.
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
C      write(*,'(1X,A,F12.5,A,F12.5)') 'SETPOINT 16 CHANGE:',
C     +                SETPT(16),'-->',SETPT(16) + DXMV  * 130. / 100.
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
C       G=4651207995.D0
C       d00_tr_new: G=5687912315.D0       
C      original: G=1431655765.D0
C        d00_tr: G=4243534565.D0
C        d01_tr: G=7854912354.D0
C        d02_tr: G=3456432354.D0
C        d03_tr: G=1731738903.D0
C        d04_tr: G=4346024432.D0
C        d05_tr: G=5784921734.D0
C        d06_tr: G=6678322168.D0
C        d07_tr: G=7984782901.D0
C        d08_tr: G=8934302332.D0
C        d09_tr: G=9873223412.D0
C        d10_tr: G=1089278833.D0
C        d11_tr: G=1940284333.D0
C        d12_tr: G=2589274931.D0
C        d13_tr: G=3485834345.D0
C        d14_tr: G=4593493842.D0
C        d15_tr: G=5683213434.D0
C        d16_tr: G=6788343442.D0
C        d17_tr: G=1723234455.D0
C        d18_tr: G=8943243993.D0
C       dd18_tr: G=1234567890.D0

C        d19_tr: G=9445382439.D0
C        d20_tr: G=9902234324.D0
C        d21_tr: G=2144342545.D0
C        d22_tr: G=3433249064.D0
C        d23_tr: G=4356565463.D0
C        d24_tr: G=8998485332.D0
C        d25_tr: G=7654534567.D0
C        d26_tr: G=5457789234.D0
C
C        d00_te: G=1254545354.D0
C        d01_te: G=2994833239.D0
C        d02_te: G=2891123453.D0
C        d03_te: G=3420494299.D0
C        d04_te: G=4598956239.D0
C        d05_te: G=5658678765.D0
C        d06_te: G=6598593453.D0
C        d07_te: G=7327843434.D0
C        d08_te: G=8943242344.D0
C        d09_te: G=9343430004.D0
C        d10_te: G=1039839281.D0
C        d11_te: G=1134345551.D0
C        d12_te: G=2232323236.D0
C        d13_te: G=3454354353.D0
C        d14_te: G=4545445883.D0
C        d15_te: G=5849489384.D0
C        d16_te: G=6284545932.D0
C        d17_te: G=4342232344.D0
C        d18_te: G=5635346588.D0
C        d19_te: G=9090909232.DO
C        d20_te: G=8322308324.D0
C        d21_te: G=2132432423.D0
C        d22_te: G=5454589923.D0
C        d23_te: G=6923255678.D0
C        d24_te: G=8493323434.D0
C        d25_te: G=9338398429.D0
C        d26_te: G=1997072199.D0


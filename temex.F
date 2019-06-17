#include "fintrf.h"
C======================================================================
C     Computational subroutine in Fortran

      subroutine convec(lin,col,dynmat)
      implicit none
      logical l1, l2
      integer i, j, i1, i2, lin, col
      real*8 dynmat(lin,col)

      do i=1,lin
        do j=1,col
          dynmat(i,j) = i*j
        enddo
      enddo
      return
      end

C-----------------------------------------------------------------------
C     Gateway routine
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)

C     Declarations
      implicit none

C     mexFunction arguments:
      mwPointer plhs(*), prhs(*)
      integer nlhs, nrhs

C     Function declarations:
      mwPointer mxGetPr
      mwPointer mxCreateDoubleMatrix, mxCreateDoubleScalar

      real*8, allocatable :: dynmat(:,:) ! 2-D dynamically allocatable array
      integer lin, col
      
      lin = 4
      col = 7

      allocate( dynmat(lin,col) )

      
      
C-----------------------------------------------------------------------
C     Check for proper number of arguments. 
      if( .false. ) then

      if (nrhs .ne. 1) then
         call mexErrMsgIdAndTxt ('MATLAB:convec:nInput',
     +                           'One input required.')
      elseif (nlhs .gt. 1) then
         call mexErrMsgIdAndTxt ('MATLAB:convec:nOutput',
     +                           'Too many output arguments.')
      endif

      endif

C     Call the computational subroutine.
      call convec(lin, col, dynmat)

      plhs(1) = mxCreateDoubleMatrix(lin,col, 0)
      call mxCopyReal8ToPtr(dynmat,mxGetPr(plhs(1)),lin*col)

      deallocate( dynmat )
      return
      end



!-------------------------------------------------------------------------------
! write vtm & vtr files, output in csv format
! finally, this program will input the output from Nerual Networks model, 

! author        Date            Version
! Xia Shuning   2024.11.19      v0.1
!-------------------------------------------------------------------------------

program write_vtk
  implicit none
  integer, parameter :: FILE_CSV_U=1000, FILE_VTR_U=1001
  integer :: i, j, k
  integer :: XNum, YNum, ZNum
  real(8), dimension(:,:,:), allocatable :: CellVolume, P, U, V, W
  real(8), dimension(:), allocatable :: X, Y, Z

  print *, "Please enter XNum, YNum, & ZNum: "
  read(*, *) XNum, YNum, ZNum

  allocate(CellVolume(1:XNum,1:YNum,1:ZNum))
  allocate(P(1:XNum,1:YNum,1:ZNum))
  allocate(U(1:XNum,1:YNum,1:ZNum))
  allocate(V(1:XNum,1:YNum,1:ZNum))
  allocate(W(1:XNum,1:YNum,1:ZNum))

  allocate(X(1:XNum))
  allocate(Y(1:YNum))
  allocate(Z(1:ZNum))

  open(FILE_CSV_U, file="output.csv", form="formatted", action="read", status="old")

  ! omit the head
  read(FILE_CSV_U,*)


  do k = 1, ZNum
    do j = 1, YNum
      do i = 1, XNum
        write(FILE_CSV_U, "(8F16.8)") X(i), Y(j), Z(K), CellVolume(i,j,k), P(i,j,k), U(i,j,k), V(i,j,k), W(i,j,k)
      end do
    end do
  end do

  open(unit=FILE_VTR_U, file  ="test.vtr",  &
                        status="old",       &
                        action="read",      &
                        access="stream",    &
                        form  ="unformatted")

  
  close(FILE_VTR_U)

  close(FILE_CSV_U)
end program write_vtk
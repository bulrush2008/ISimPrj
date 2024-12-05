
!-------------------------------------------------------------------------------
! read vtm & vtr files, output in csv format

! author        Date            Version
! Xia Shuning   2024.11.19      v1.0
!-------------------------------------------------------------------------------

program read_vtk
  implicit none
  integer, parameter :: FILE_VTM_U=1000, FILE_VTR_U=1001, FILE_CSV_U=1002
  character(len=100) :: buffer
  integer :: blockNum
  character(len=100) :: VTRFileName
  integer :: is,ie, js,je, ks,ke 
  integer :: thePos, offset_var, offset_x, offset_y!, offset_z
  integer :: XNum, YNum, ZNum, varNum
  integer :: i, j, k
  integer, dimension(1:8) :: offsets

  real(8), dimension(:,:,:), allocatable :: CellVolume, P, U, V, W
  real(8), dimension(:), allocatable :: X, Y, Z

  !*****************************************************************************
  ! var           | mean
  ! offsets       | offsets for each variable
  !               | 1 for 'CellVolume'
  !               | 2 for 'P'
  !               | 3 for 'U'
  !               | 4 for 'V'
  !               | 5 for 'W'
  !               | 6 for 'Xcenter'
  !               | 7 for 'Ycenter'
  !               | 8 for 'Zcenter'
  !*****************************************************************************

  open(unit=FILE_VTM_U, file  ="./RESULT/ld3d.000200.vtm",  &
                        status="old",                       &
                        access="stream",                    &
                        action="read",                      &
                        form  ="unformatted")

  read(FILE_VTM_U) buffer(1:22)
  !print *, buffer(1:21)

  read(FILE_VTM_U) buffer(1:78)
  !print *, buffer(1:77)

  read(FILE_VTM_U) buffer(1:25)
  !print *, buffer(1:24)

  read(FILE_VTM_U) buffer(1:50)
  !print *, buffer(1:49)! <DataSet index="   1" file="ld3d/200/1.vtr"/>

  read(buffer(21:24), "(I4)") blockNum
  !print *, blockNum

  VTRFileName = buffer(33:46)
  !print *, VTRFileName

  VTRFileName = "./RESULT/"//trim(adjustl(VTRFileName))

  open(unit=FILE_VTR_U, file  =VTRFileName, &
                        status="old",       &
                        action="read",      &
                        access="stream",    &
                        form  ="unformatted")

  read(FILE_VTR_U) buffer(1:22)
  !print *, buffer(1:21)

  read(FILE_VTR_U) buffer(1:73)
  !print *, buffer(1:72)

  read(FILE_VTR_U) buffer(1:65)
  !print *, buffer(1:64)

  !read(buffer(33:37),"(I5)") is
  !read(buffer(38:42),"(I5)") ie
  !read(buffer(43:47),"(I5)") js
  !read(buffer(48:52),"(I5)") je
  !read(buffer(53:57),"(I5)") ks
  !read(buffer(58:62),"(I5)") ke

  read(buffer(33:62),"(6I5)") is, ie, js, je, ks, ke
  !print *, is, ie, js, je, ks, ke

  XNum = ie-is+1
  YNum = je-js+1
  ZNum = ke-ks+1

  allocate(CellVolume(1:XNum,1:YNum,1:ZNum))
  allocate(P(1:XNum,1:YNum,1:ZNum))
  allocate(U(1:XNum,1:YNum,1:ZNum))
  allocate(V(1:XNum,1:YNum,1:ZNum))
  allocate(W(1:XNum,1:YNum,1:ZNum))

  allocate(X(1:XNum))
  allocate(Y(1:YNum))
  allocate(Z(1:ZNum))

  offset_var = (ie-is+1)*(je-js+1)*(ke-ks+1) * 8 + 4

  offsets(1) = 0  ! 'CellVolume'
  offsets(2) = offsets(1) + offset_var  ! "P"
  offsets(3) = offsets(2) + offset_var  ! "U"
  offsets(4) = offsets(3) + offset_var  ! "V"
  offsets(5) = offsets(4) + offset_var  ! "W"

  offsets(6) = offsets(5) + offset_var  ! "Xcenter"

  offset_x = (ie-is+1)*8 + 4
  offsets(7) = offsets(6) + offset_x  ! "Ycenter"

  offset_y = (je-js+1)*8 + 4
  offsets(8) = offsets(7) + offset_y

  !print *, "offsets = ", offsets

  inquire(FILE_VTR_U, pos=thePos)
  read(FILE_VTR_U, pos=thePos+974) buffer
  !print *, buffer

  inquire(FILE_VTR_U, pos=thePos)
  !print *, "The position at begin of many body: ", thePos

  ! read 'CellVolume'
  read(FILE_VTR_U, pos=thePos+1) varNum, (((CellVolume(i,j,k),i=1,XNum),j=1,YNum),k=1,ZNum)
  !print *, varNum
  !print *, CellVolume(1:5,1,1)

  ! read 'P'
  read(FILE_VTR_U) varNum, (((P(i,j,k),i=1,XNum),j=1,YNum),k=1,ZNum)
  !print *, varNum
  !print *, "P= ", P(1:5,4,4)

  read(FILE_VTR_U) varNum, (((U(i,j,k),i=1,XNum),j=1,YNum),k=1,ZNum)
  !print *, varNum
  !print *, "U= ", U(1:5,4,4)

  read(FILE_VTR_U) varNum, (((V(i,j,k),i=1,XNum),j=1,YNum),k=1,ZNum)
  !print *, varNum
  !print *, "V= ", V(1:5,4,4)

  read(FILE_VTR_U) varNum, (((W(i,j,k),i=1,XNum),j=1,YNum),k=1,ZNum)
  !print *, varNum
  !print *, "W= ", W(1:5,4,4)

  read(FILE_VTR_U) varNum, (X(i),i=1,XNum)
  read(FILE_VTR_U) varNum, (Y(j),j=1,YNum)
  read(FILE_VTR_U) varNum, (Z(k),k=1,ZNum)

  !print *, "X= ", X
  !print *, "Y= ", Y
  !print *, "Z= ", Z

  ! write to file: .csv
  open(FILE_CSV_U, file="output.csv", action="write")
  write(FILE_CSV_U, "(A)") "X, Y, Z, CellVolume, P, U, V, W"

  do k = 1, ZNum
    do j = 1, YNum
      do i = 1, XNum
        write(FILE_CSV_U, "(8F16.8)") X(i), Y(j), Z(K), CellVolume(i,j,k), P(i,j,k), U(i,j,k), V(i,j,k), W(i,j,k)
      end do
    end do
  end do

  close(FILE_CSV_U)
  close(FILE_VTR_U)
  close(FILE_VTM_U)
end program read_vtk
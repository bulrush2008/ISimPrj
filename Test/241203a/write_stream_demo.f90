
program main
  implicit none
  character(len=50) :: buffer
  integer, parameter :: Nx=4, Ny=5
  integer :: i,j
  real(8), dimension(:,:), allocatable :: field

  allocate(field(1:Nx,1:Ny))

  do j = 1,Ny
    do i = 1,Nx
      field(i,j) = real(i+10*j) + 0.1
    end do
  end do

  open(unit=1001, file="demo.dat", status="replace", form="unformatted", access="stream")

  !print*, "debug, line 17"
  write(buffer,"(a30,i4)") "demo file header ", 100

  write(1001) trim(adjustl(buffer))//char(10)

  write(buffer,"(a)") "_"
  write(1001) trim(buffer)

  write(1001) Nx*Ny, ((field(i,j),i=1,Nx),j=1,Ny)

  close(1001)

  deallocate(field)
end program main
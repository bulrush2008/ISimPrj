
program main
  implicit none
  character(len=50) :: buffer

  open(1001, file='demo.dat', access='stream', status='replace', form='unformatted')
  write(buffer, '(a)') '_'
  write(1001) trim(buffer)
  print *, trim(buffer)
  close(1001)
end program main
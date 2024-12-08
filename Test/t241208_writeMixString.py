
vtm = open('t01.vtm', 'wb')

import io
#mStr = <?xml version="1.0"?>

f = io.BytesIO(b'<?xml version="1.0"?>')

#print(f.getvalue().decode("ASCII"))

vtm.write(f.getvalue())
vtm.write(b"\n")
vtm.write(b'<?xml version="1.0"?>')


vtm.close()
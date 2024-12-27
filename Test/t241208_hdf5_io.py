
"""Create an HDF5 file in memory and retrieve the raw bytes

This could be used, for instance, in a server producing small HDF5
files on demand.
"""
import io
import h5py

bio = io.BytesIO()
with h5py.File(bio, 'w') as f:
    f['dataset'] = range(10)

data = bio.getvalue() # data is a regular Python bytes object.
print("Total size:", len(data))
print("First bytes:", data[:10])

import io

# writes to a buffer
output = io.StringIO()

output.write("This goes into the buffer.")

print("And so does this.", file=output)

# retrieve the value written
print(output.getvalue())

output.write("%04d\n"%5)  # note "\n"
print(output.getvalue())

output.write("%16.7f"%3.25)
print(output.getvalue())

output.close()  # Discard buffer memory
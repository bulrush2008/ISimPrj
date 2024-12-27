
import pathlib

usr = pathlib.PurePosixPath("/usr")
print(usr, type(usr))

usr_local = usr/"local"; print(usr_local)

usr_share = usr / pathlib.PurePosixPath("share"); print(usr_share)

root = usr / ".."; print(root)
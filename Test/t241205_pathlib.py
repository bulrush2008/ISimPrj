
from pathlib import Path

dir = Path("../FSCases")
print(dir)

fullDir = dir.joinpath("case1_point.002000.vtm")
print(fullDir)
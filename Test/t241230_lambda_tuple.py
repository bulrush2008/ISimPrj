
a = [2,28,2,53,2,13]

f = lambda l: (l[1]-l[0]+1, l[3]-l[2]+1, l[5]-l[4]+1)

t = f(a); print(t)

g = lambda t: t[0]*t[1]*t[2]
s = g(t); print(s, "?", 27*52*12)
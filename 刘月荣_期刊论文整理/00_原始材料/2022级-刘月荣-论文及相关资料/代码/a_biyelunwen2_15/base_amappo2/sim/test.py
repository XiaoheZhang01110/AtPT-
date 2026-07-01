base = 100
pax = []
list1 = [0,0,0,3,0,6,0,0,1]
for i in range(len(list1)):
    if list1[i] > 0:
        pax+=[base-i for t in range(list1[i])]

print(pax)


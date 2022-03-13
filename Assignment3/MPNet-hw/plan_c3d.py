import numpy as np

def IsInCollision(x,obc):
    size = [[5, 5, 10], [5, 10, 5], [5, 10, 10], [10, 5, 5], [10, 5, 10], [
        10, 10, 5], [10, 10, 10], [5, 5, 5], [10, 10, 10], [5, 5, 5]]
    s=np.zeros(3,dtype=np.float32)
    s[0]=x[0] # point x coord
    s[1]=x[1] # point y coord
    s[2]=x[2] # point z coord
    for i in range(0, len(obc)): # for 10 obstacles
        colliding=True
        for j in range(0,3):
            if abs(obc[i][j] - s[j]) > size[i][j]/2.0 and s[j]<20.0 and s[j]>-20:
                colliding=False
                break
        if colliding==True:
            return True
    return False

import numpy as np

def ema_value(rate, current, previous):
    return rate * current + (1-rate) * previous

def atan_circular(y, x):
    return np.arctan2(y,x) + -1*(np.sign(np.arctan2(y,x))-1)*np.pi

def zoomed_mantissa(a, b, f, out = None):
    x = int(np.round(a * f))
    y = int(np.round(b * f))
    v = 0
    w = 0
    r = x
    s = y
    i = 1
#
    if (x == 0 and y != 0) or (x != 0 and y == 0) or (x == y):
        return a - b
#
    a = r % 10
    b = s % 10
    v = a
    w = b
    r = int(r / 10)
    s = int(s / 10)    
#
    while r > 0 and s > 0 and r != s:        
        a = r % 10
        b = s % 10        
        v = v + a * np.power(10, i)
        w = w + b * np.power(10, i)
        r = int(r/10)
        s = int(s/10)
        i += 1
#
    if out is None:
        return np.array((v, w))
    out[0] = v
    out[1] = w

    return out
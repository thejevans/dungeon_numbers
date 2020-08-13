'''
docstring
'''
import sys
from itertools import product
import numpy as np
from tqdm import tqdm

def float_to_list(number):
    '''
    docstring
    '''
    str_num = str(number)

    num_list = str_num.split('.')
    sign = 1
    if len(num_list) > 1:
        pre = num_list[0]
        post = num_list[1]
    else:
        pre = num_list[0]
        post = '0'

    if str_num[0] == '-':
        pre = pre[1:]
        sign = -1

    pre  = [float(c) for c in pre[::-1]]
    post = [float(c) for c in post[:]]

    return [sign, pre, post]

def convert_to_base_10(real, imag, radix):
    '''
    Takes a complex number in a given complex base (radix) and returns it in base-10

    parameters:
        real: a list representation of the real component of a complex number
        imag: a list representation of the imaginary comonent of a complex number
        radix: the complex radix that the given number is assumed to be in the base of

    returns:
        the input complex number converted from base-radix to base-10
    '''

    real_sign = real[0]
    real_pre  = real[1]
    real_post = real[2]

    imag_sign = imag[0]
    imag_pre  = imag[1]
    imag_post = imag[2]

    radix_perp = -radix.imag + radix.real*1j

    out = 0 + 0j

    # convert to base-10
    for i, val in enumerate(real_pre):
        out += val * (radix**i)

    for i, val in enumerate(real_post):
        out += val * 1/(radix**(i+1))

    for i, val in enumerate(imag_pre):
        out += val * (radix_perp**i)

    for i, val in enumerate(imag_post):
        out += val * 1/(radix_perp**(i+1))

    return out.real * real_sign + out.imag * imag_sign * 1j

n = int(sys.argv[1])
m = int(sys.argv[2])
lo = float(sys.argv[3])
hi = float(sys.argv[4])
outfile = sys.argv[5]

X, Y   = np.meshgrid(np.linspace(lo, hi, n), np.linspace(lo, hi, n))
output = np.zeros((100, n, n), dtype=np.complex)

output[0, :, :] = X + Y*1j

x = [[float_to_list(X[i, j]) for j in range(n)] for i in range(n)]
y = [[float_to_list(Y[i, j]) for j in range(n)] for i in range(n)]

for i, j in tqdm(product(range(n), range(n)), total=n*n):
    for _ in range(m-99):
        output[0, i, j]   = convert_to_base_10(x[i][j], y[i][j], output[0, i, j])
    for k in range(99):
        output[k+1, i, j] = convert_to_base_10(x[i][j], y[i][j], output[k, i, j])

np.save(outfile, output)

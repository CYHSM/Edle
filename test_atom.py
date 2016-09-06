# %matplotlib inline
#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# mpl.style.use('ggplot')
# mpl.style.available
#
# x = np.linspace(0, 10, 500)
# x = 2 * x
import multiprocessing as mp
import time
def foo(q):
    print('a')
    time.sleep(5)
    print('b')
    q.put('hello')
    print('c')

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    p.start()
    print('1')
    print(q.get())
    print('2')
    p.join()
    print('3')

import time

from random import randrange
from multiprocessing.pool import ThreadPool

from tqdm import tqdm


def func_call(position, total):
    text = 'progressbar #{position}'.format(position=position)
    with  tqdm(total=total, position=position, desc=text) as progress:
        for _ in range(0, total, 5):
            progress.update(5)
            time.sleep(randrange(3))


pool = ThreadPool(10)
tasks = range(5)
for i, url in enumerate(tasks, 1):
    pool.apply_async(func_call, args=(i, 100))
pool.close()
pool.join()
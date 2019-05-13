""" This module introduces tools for execution of jobs in parallel.

Per default, joblib is used for easy and efficient execution of parallel tasks.
However, joblib does not support GPU management, and does not kill processes
at the end of each task, thus keeping in GPU memory the pytorch execution
context.

This module introduces equivalent tools for multiprocessing while
avoiding GPU memory leak.


.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""

import multiprocessing as mp
from multiprocessing import Manager
from time import sleep
import os
import signal
from .Settings import SETTINGS


def worker_subprocess(function, devices, lockd, results, lockr,
                      pids, lockp, args, kwargs, idx, *others):
        device = None
        while device is None:
            with lockd:
                try:
                    device = devices.pop()
                except IndexError:
                    pass
            sleep(2)
        with lockp:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    pids.remove(pid)
                except ProcessLookupError:
                    pass
        output = function(*args, **kwargs, device=device, idx=idx)
        with lockd:
            devices.append(device)
        with lockr:
            results.append(output)
        with lockp:
            pids.append(os.getpid())


def parallel_run(function, *args, nruns=1, njobs=None, gpus=None, **kwargs):
    njobs = SETTINGS.get_default(nb_jobs=njobs)
    gpus = SETTINGS.get_default(gpu=gpus)
    manager = Manager()
    devices = manager.list([f'cuda:{i%gpus}' if gpus !=0
                            else 'cpu' for i in range(njobs)])
    results = manager.list()
    pids = manager.list()
    lockd = manager.Lock()
    lockr = manager.Lock()
    lockp = manager.Lock()
    poll = [mp.Process(target=worker_subprocess,
                       args=(function, devices,
                             lockd, results, lockr,
                             pids, lockp, args,
                             kwargs, i))
            for i in range(nruns)]
    for p in poll:
        p.start()
    for p in poll:
        p.join()

    return list(results)

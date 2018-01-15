#!/usr/bin/env python2

""" Collect info on nvidia gpu and display graph

requires nvidia-ml-py, can be installed via pip (currently only py2)

TODO: remember data, then provide it some way to other program for (continuous)
      visualization

Author: Christian Herdtweck
"""

from __future__ import print_function

import sys
from argparse import ArgumentParser
import logging
from time import sleep
from datetime import time, timedelta, datetime as dt
import re
from subprocess import call

try:
    from pynvml import *
except ImportError:
    print('nvidia-ml-py not installed', file=sys.stderr)
    sys.exit(1)


# pattern for --shutdown argument
SHUT_ARGS_PATTERN = r'(?P<start_h>\d{1,2}):(?P<start_min>\d{2})-' + \
                    r'(?P<end_h>\d{1,2}):(?P<end_min>\d{2}),' + \
                    r'(?P<idle_duration>\d+),' + \
                    r'(?P<idle_percent>\d{1,2})$'

SHUT_ARGS_DEFAULT = '1:30-9:00,30,10'


def parse_args(args=None):
    """ handle command line arguments, sys.argv per default """
    parser = ArgumentParser()
    parser.add_argument('-i', '--interval', type=float, default=1,
                        help='refresh interval in seconds')
    parser.add_argument('-g', '--gpu-index', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='produce more logging output')
    parser.add_argument('-s', '--shutdown', type=str,
                        default=SHUT_ARGS_DEFAULT,
                        help='enable auto-shutdown when gpu is inactive for '
                             'a longer time. Format: '
                             'start-end;duration;percent, where start,end are '
                             'H[H[:MM]], duration is minutes [int], and '
                             'percent is int')
    parser.add_argument('-l', '--log-file', default='', type=str,
                        help='Log to given file; if no file given (default), '
                             'log to stdout')
    args = parser.parse_args(args)

    # parse shutdown args
    if args.shutdown:
        shut_args = re.match(SHUT_ARGS_PATTERN, args.shutdown)
        if not shut_args:
            print('failed to parse shutdown args {0!r}'.format(args.shutdown),
                  file=sys.stderr)
            return 2
        shut_args = shut_args.groupdict()
        shut_args['start'] = time(int(shut_args['start_h']),
                                  int(shut_args['start_min']))
        shut_args['end'] = time(int(shut_args['end_h']),
                                int(shut_args['end_min']))
        shut_args['idle_duration'] = \
                timedelta(seconds=60*int(shut_args['idle_duration']))
        shut_args['idle_percent'] = int(shut_args['idle_percent'])
    else:
        shut_args = None

    return args, shut_args


def get_data(handle):
    """ run nvidia-smi, parse data from it
    
    assumes nvmlInit() has been run

    return %gpu, %mem,
    """
    util = nvmlDeviceGetUtilizationRates(handle)
    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    fan = nvmlDeviceGetFanSpeed(handle)
    result = dict(gpu=util.gpu, mem=util.memory, temp=temp, fan=fan,
                  time=dt.now())
    return result


def print_data(data, will_shut, log):
    """ nice readable printing of result from get_data """
    shutstr = ', will shut down at {0:%H:%M:%S}'.format(will_shut) \
            if will_shut else ''
    msg = '{0[time]:%H:%M:%S}: gpu {0[gpu-idx]} at {0[gpu]:02d}%, ' \
          'mem at {0[mem]:02d}%, temp {0[temp]:02d}C {1}' \
          .format(data, shutstr)
    log.info(msg)


def check_shutdown(shut_args, gpu_usage, planned_shut, log):
    """ check if gpu has been idle for long enough to shut down """

    if gpu_usage > shut_args['idle_percent']:
        log.debug('no shutdown, gpu busy')
        if planned_shut is not None:
            log.info('GPU now busy again, shutdown aborted')
        return None
    now_full = dt.now()
    now = now_full.time()
    if shut_args['start'] < shut_args['end'] and not \
            shut_args['start'] < now < shut_args['end']:
        log.debug('no shutdown, outside range1')
        if planned_shut is not None:
            log.info('time now outside range1, shutdown aborted')
        return None
    elif shut_args['start'] > shut_args['end'] and \
            (now < shut_args['start'] and now > shut_args['end']):
        log.debug('no shutdown, outside range2')
        if planned_shut is not None:
            log.info('time now outside range2, shutdown aborted')
        return None

    # now we are in shutdown range and gpu is not busy
    if planned_shut is None:
        log.info('GPU now idle, will shutdown in {0} if this stays'
                 .format(shut_args['idle_duration']))
        return now_full + shut_args['idle_duration']  # we just start being idle
    if now_full < planned_shut:
        log.debug('no shutdown, {0} left'
                  .format(planned_shut - now_full))
        return planned_shut

    # do shutdown
    log.warning('been idle long enough, shutting down now')
    do_shutdown()


def do_shutdown():
    call(['systemctl', 'suspend'])


def main(args=None):
    """ main function, called when running this as script """
    args, shut_args = parse_args(args)
    log = logging.getLogger('nvidia-mon')
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if args.log_file:
        log.addHandler(logging.FileHandler(args.log_file))
    else:
        log.addHandler(logging.StreamHandler(sys.stdout))
    log.info('start logging')
    log.debug('shut args: {0}'.format(shut_args))
    will_shut = None

    try:
        log.debug('initializing nvml')
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        log.debug("found %i gpus" % device_count)
        while True:
            handle = nvmlDeviceGetHandleByIndex(args.gpu_index)
            data = get_data(handle)
            data['gpu-idx'] = args.gpu_index
            if shut_args:
                will_shut = check_shutdown(shut_args, data['gpu'], will_shut,
                                            log)
            print_data(data, will_shut, log)
            sleep(args.interval)
    except Exception:
        raise
    finally:
        log.debug('shutting down nvml')
        nvmlShutdown()
    
    log.debug('done.')
    logging.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())

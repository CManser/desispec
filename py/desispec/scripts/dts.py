# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.scripts.dts
====================

Entry point for :command:`desi_dts`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import datetime as dt
import os
import subprocess as sub
import sys
import time
from collections import namedtuple
from desiutil.log import get_logger, DEBUG


log = get_logger(timestamp=True)


DTSDir = namedtuple('DTSDir', 'source, staging, destination, hpss')


config = [DTSDir('/data/dts/exposures/raw',
                 os.path.realpath(os.path.join(os.environ['DESI_ROOT'], 'spectro', 'staging', 'raw')),
                 os.path.realpath(os.environ['DESI_SPECTRO_DATA']),
                 'desi/spectro/data'),]


def pack_args(options):
    """Parse and format NERSC-specific command-line options.

    Parameters
    ----------
    :class:`argparse.Namespace`
        The parsed command-line options.

    Returns
    -------
    :class:`list`
        Command-line options that can be appended to an existing command.
    """
    optlist = ("nersc",
               "nersc_queue",
               "nersc_queue_redshifts",
               "nersc_maxtime",
               "nersc_maxnodes",
               "nersc_maxnodes_small",
               "nersc_maxnodes_redshifts",
               "nersc_shifter",
               "mpi_procs",
               "mpi_run",
               "procs_per_node")
    varg = vars(options)
    opts = list()
    for k, v in varg.items():
        if k in optlist:
            if v is not None:
                opts.append("--{0}".format(k))
                if not isinstance(v, bool):
                    opts.append(v)
    return opts


def _options(*args):
    """Parse command-line options for DTS script.

    Parameters
    ----------
    args : iterable
        Arguments to the function will be parsed for testing purposes.

    Returns
    -------
    :class:`argparse.Namespace`
        The parsed command-line options.
    """
    from argparse import ArgumentParser
    desc = "Transfer DESI raw data files."
    prsr = ArgumentParser(prog=os.path.basename(sys.argv[0]), description=desc)
    prsr.add_argument('-b', '--backup', metavar='H', type=int, default=20,
                      help='UTC time in hours to trigger HPSS backups (default %(default)s:00 UTC).')
    prsr.add_argument('-d', '--debug', action='store_true',
                      help='Set log level to DEBUG.')
    prsr.add_argument('-k', '--kill', metavar='FILE',
                      default=os.path.join(os.environ['HOME'], 'stop_dts'),
                      help="Exit the script when FILE is detected (default %(default)s).")
    prsr.add_argument('-n', '--nersc', default='cori', metavar='NERSC_HOST',
                      help="Trigger DESI pipeline on this NERSC system (default %(default)s).")
    prsr.add_argument('-P', '--no-pipeline', action='store_false', dest='pipeline',
                      help="Only transfer files, don't start the DESI pipeline.")
    # prsr.add_argument('-p', '--prefix', metavar='PREFIX', action='append',
    #                   help="Prepend one or more commands to the night command.")
    prsr.add_argument('-s', '--sleep', metavar='M', type=int, default=10,
                      help='Sleep M minutes before checking for new data (default %(default)s minutes).')
    # prsr.add_argument('filename', metavar='FILE',
    #                   help='Filename with path of delivered file.')
    # prsr.add_argument('exposure', type=int, metavar='EXPID',
    #                   help='Exposure number.')
    # prsr.add_argument('night', metavar='YYYYMMDD', help='Night ID.')
    # prsr.add_argument('nightStatus',
    #                   choices=('start', 'update', 'end'),
    #                   help='Start/end info.')
    if len(args) > 0:
        options = prsr.parse_args(args)
    else:  # pragma: no cover
        options = prsr.parse_args()
    return options


def check_exposure(dst, expid):
    """Ensure that all files associated with an exposure have arrived.

    Parameters
    ----------
    dst : :class:`str`
        Delivery directory, typically ``DESI_SPECTRO_DATA/NIGHT``.
    expid : :class:`int`
        Exposure number.

    Returns
    -------
    :class:`bool`
        ``True`` if all files have arrived.
    """
    files = ('fibermap-{0:08d}.fits', 'desi-{0:08d}.fits.fz', 'guider-{0:08d}.fits.fz')
    return all([os.path.exists(os.path.join(dst, f.format(expid))) for f in files])


def move_file(filename, dst):
    """Move delivered file from the DTS spool to the final raw data area.

    This function will ensure that the destination directory exists.

    Parameters
    ----------
    filename : :class:`str`
        The name, including full path, of the file to move.
    dst : :class:`str`
        The destination *directory*.

    Returns
    -------
    :class:`str`
        The value returned by :func:`shutil.move`.
    """
    from shutil import move
    if not os.path.exists(dst):
        log.info("mkdir('{0}', 0o2770)".format(dst))
        os.mkdir(dst, 0o2770)
    log.info("move('{0}', '{1}')".format(filename, dst))
    return move(filename, dst)


def main():
    """Entry point for :command:`desi_dts`.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    global log
    options = _options()
    if options.debug:
        log.setLevel(DEBUG)
    ssh = 'ssh'
    # ssh = '/bin/ssh'
    while True:
        log.info('Starting transfer loop.')
        if os.path.exists(options.kill):
            log.info("%s detected, shutting down transfer daemon.", options.kill)
            return 0
        #
        # Find symlinks at KPNO.
        #
        for d in config:
            status_dir = os.path.join(os.path.dirname(d.staging), 'status')
            cmd = [ssh, '-q', 'dts', '/bin/find', d.source, '-type', 'l']
            log.debug(' '.join(cmd))
            p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
            out, err = p.communicate()
            links = sorted([x for x in out.decode('utf-8').split('\n') if x])
            if links:
                for l in links:
                    exposure = os.path.basename(l)
                    night = os.path.basename(os.path.dirname(l))
                    #
                    # New night detected?
                    #
                    n = os.path.join(d.staging, night)
                    if not os.path.isdir(n):
                        log.debug("os.makedirs('%s', exist_ok=True)", n)
                        # os.makedirs(n, exist_ok=True)
                    #
                    # Has exposure already been transferred?
                    #
                    se = os.path.join(n, exposure)
                    de = os.path.join(d.destination, night, exposure)
                    if not os.path.isdir(se) and not os.path.isdir(de):
                        cmd = ['/bin/rsync', '--verbose', '--no-motd',
                               '--recursive', '--copy-dirlinks', '--times',
                               '--omit-dir-times',
                               'dts:'+os.path.join(d.source, night, exposure)+'/',
                               se+'/']
                        log.debug(' '.join(cmd))
                        # p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
                        # out, err = p.communicate()
                        # status = str(p.returncode)
                        status = '0'
                    else:
                        log.info('%s already transferred.', se)
                        status = 'done'
                    if status == '0':
                        pass
                    elif status == 'done':
                        #
                        # Do nothing, successfully.
                        #
                        pass
                    else:
                        log.error('rsync problem detected!')
            else:
                log.warning('No links found, check connection.')
            #
            # Are any nights eligible for backup?
            # 12:00 MST = 19:00 UTC.
            # Plus one hour just to be safe, so after 20:00 UTC.
            #
            yesterday = (dt.datetime.now() - dt.timedelta(seconds=86400)).strftime('%Y%m%d')
            now = int(dt.datetime.utcnow().strftime('%H'))
            hpss_file = d.hpss.replace('/', '_')
            ls_file = os.path.join(os.environ['CSCRATCH'], hpss_file + '.txt')
            if now >= options.backup:
                pass
        # time.sleep(options.sleep*60)
        time.sleep(options.sleep)
    return 0

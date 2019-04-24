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
import shutil
import stat
import subprocess as sub
import sys
import time
from collections import namedtuple
from desiutil.log import get_logger, DEBUG


log = get_logger(timestamp=True)


DTSDir = namedtuple('DTSDir', 'source, staging, destination, hpss')


dir_perm  = (stat.S_ISGID |
             stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
             stat.S_IRGRP | stat.S_IXGRP)
file_perm = stat.S_IRUSR | stat.S_IRGRP


expected_files = ('desi-{exposure}.fits.fz',
                  'fibermap-{exposure}.fits',
                  'guider-{exposure}.fits.fz')


desi_night = os.path.realpath(os.path.join(os.environ['HOME'], 'bin',
                                           'wrap_desi_night.sh'))


def _config():
    """Wrap configuration so that module can be imported without
    environment variables set.
    """
    return [DTSDir('/data/dts/exposures/raw',
                   os.path.realpath(os.path.join(os.environ['DESI_ROOT'], 'spectro', 'staging', 'raw')),
                   os.path.realpath(os.environ['DESI_SPECTRO_DATA']),
                   'desi/spectro/data'),]


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


def check_exposure(destination, exposure):
    """Ensure that all files associated with an exposure have arrived.

    Parameters
    ----------
    destination : :class:`str`
        Delivery directory, typically ``DESI_SPECTRO_DATA/NIGHT``.
    exposure : :class:`str`
        Exposure number.

    Returns
    -------
    :class:`bool`
        ``True`` if all files have arrived.
    """
    return all([os.path.exists(os.path.join(destination,
                                            f.format(exposure=exposure)))
                for f in expected_files])


def verify_checksum(checksum_file, files):
    """Verify checksums supplied with the raw data.

    Parameters
    ----------
    checksum_file : str
        The checksum file.
    files : list
        The list of files in the directory containing the checksum file.

    Returns
    -------
    int
        An integer that indicates the number of checksum mismatches.
    """
    with open(checksum_file) as c:
        data = c.read()
    lines = data.split('\n')
    errors = 0
    if len(lines) == len(files):
        digest = dict([(l.split()[1], l.split()[0]) for l in lines if l])
        d = os.path.dirname(checksum_file)
        for f in files:
            ff = os.path.join(d, f)
            if ff != checksum_file:
                with open(ff, 'rb') as fp:
                    h = hashlib.sha256(fp.read).hexdigest()
            if digest[f] == h:
                log.debug("%f is valid.", ff)
            else:
                log.error("Checksum mismatch for %s!", ff)
                errors += 1
        return errors
    else:
        log.error("%s does not match the number of files!", checksum_file)
        return -1


def pipeline_update(pipeline_host, night, exposure, command='update', ssh='ssh',
                    queue='realtime'):
    """Generate a ``desi_night`` command to pass to the pipeline.

    Parameters
    ----------
    pipeline_host : str
        Run the pipeline on this NERSC system.
    night : str
        Night of observation.
    exposure : str
        Exposure number.
    command : str, optional
        Specific command to pass to ``desi_night``.
    ssh : str, optional
        SSH command to use.
    queue : str, optional
        NERSC queue to use.

    Returns
    -------
    list
        A command suitable for passing to :class:`subprocess.Popen`.
    """
    return [ssh, '-q', pipeline_host,
            desi_night, command,
            '--night', night,
            '--expid', exposure,
            '--nersc', pipeline_host,
            '--nersc_queue', queue,
            '--nersc_maxnodes', '25']


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
        for d in _config():
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
                    #
                    # Transfer complete.
                    #
                    if status == '0':
                        #
                        # Check permissions.
                        #
                        log.debug("os.chmod('%s', 0%o)", se, dir_perm)
                        # os.chmod(se, dir_perm)
                        exposure_files = os.listdir(se)
                        for f in exposure_files:
                            ff = os.path.join(se, f)
                            if os.path.isfile(ff):
                                log.debug("os.chmod('%s', 0%o)", ff, file_perm)
                                # os.chmod(ff, file_perm)
                            else:
                                log.warning("Unexpected file type detected: %s", ff)
                        #
                        # Verify checksums.
                        #
                        checksum_file = os.path.join(se, "checksum-{0}-{1}.sha256sum".format(night, exposure))
                        if os.path.exists(checksum_file):
                            checksum_status = verify_checksum(checksum_file, exposure_files)
                        else:
                            log.warning("No checksum file for %s/%s!", night, exposure)
                            checksum_status = 0
                        #
                        # Did we pass checksums?
                        #
                        if checksum_status == 0:
                            #
                            # Set up DESI_SPECTRO_DATA.
                            #
                            dn = os.path.join(d.destination, night)
                            if not os.path.isdir(dn):
                                log.debug("os.makedirs('%s', exist_ok=True)", dn)
                                # os.makedirs(dn, exist_ok=True)
                            #
                            # Move data into DESI_SPECTRO_DATA.
                            #
                            if not os.path.isdir(de):
                                log.debug("shutil.move('%s', '%s')", se, dn)
                                # shutil.move(se, dn)
                            #
                            # Is this a "realistic" exposure?
                            #
                            if options.pipeline and check_exposure(de, exposure):
                                #
                                # Run update
                                #
                                cmd = pipeline_update(options.nersc, night, exposure)
                                log.debug(' '.join(cmd))
                                # p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
                                # out, err = p.communicate()
                                # status = str(p.returncode)
                                done_config = {'flats': 'flats', 'arcs': 'arcs',
                                               'science': 'redshifts'}
                                done = False
                                for k in done_config:
                                    if os.path.exists(os.path.join(de, '{0}-{1}-{2}.done'.format(k, night, exposure))):
                                        cmd = pipeline_update(options.nersc, night, exposure, command=done_config[k])
                                        log.debug(' '.join(cmd))
                                        # p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
                                        # out, err = p.communicate()
                                        # status = str(p.returncode)
                                        # sprun desi_dts_status --directory ${status_dir} --last $k ${night} ${exposure}
                                        done = True
                                if not done:
                                    pass
                                    # sprun desi_dts_status --directory ${status_dir} ${night} ${exposure}
                            else:
                                log.info("%s/%s appears to be test data. Skipping pipeline activation.", night, exposure)
                        else:
                            log.error("Checksum problem detected for %s/%s!", night, exposure)
                            # sprun desi_dts_status --directory ${status_dir} --failure ${night} ${exposure}
                    elif status == 'done':
                        #
                        # Do nothing, successfully.
                        #
                        pass
                    else:
                        log.error('rsync problem detected!')
                        # sprun desi_dts_status --directory ${status_dir} --failure ${night} ${exposure}
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
                if os.path.isdir(os.path.join(d.destination, yesterday)):
                    log.debug("os.remove('%s')", ls_file)
                    os.remove(ls_file)
                    cmd = ['/usr/common/mss/bin/hsi', '-O', ls_file,
                           'ls', '-l', d.hpss]
                    log.debug(' '.join(cmd))
                    # p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
                    # out, err = p.communicate()
                    # status = str(p.returncode)
                    #
                    # Both a .tar and a .tar.idx file should be present.
                    #
                    with open(ls_file) as l:
                        data = l.read()
                    backup_files = [l.split()[-1] for l in data.split('\n') if l]
                    backup_file = hpss_file + '_' + yesterday + '.tar'
                    if backup_file in backup_files and backup_file + '.idx' in backup_files:
                        log.debug("Backup of %s already complete.", yesterday)
                    else:
                        cmd = ['/usr/common/mss/bin/htar',
                               '-cvhf', d.hpss + '/' + backup_file,
                               '-H', 'crc:verify=all',
                               yesterday]
                        log.debug(' '.join(cmd))
                        # p = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE)
                        # out, err = p.communicate()
                        # status = str(p.returncode)
                else:
                    log.warning("No data from %s detected, skipping HPSS backup.", yesterday)
        # time.sleep(options.sleep*60)
        time.sleep(options.sleep)
    return 0

# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.scripts.dts
====================

Entry point for :command:`desi_dts`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import datetime as dt
import json
import logging
import os
import shutil
import stat
import subprocess as sub
import sys
import time
from collections import namedtuple
from logging.handlers import RotatingFileHandler, SMTPHandler
from socket import getfqdn
from pkg_resources import resource_filename
from desiutil.log import get_logger


log = None


DTSDir = namedtuple('DTSDir', 'source, staging, destination, hpss')


dir_perm  = (stat.S_ISGID |
             stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
             stat.S_IRGRP | stat.S_IXGRP)  # 0o2750
file_perm = stat.S_IRUSR | stat.S_IRGRP    # 0o0440


expected_files = ('desi-{exposure}.fits.fz',
                  'fibermap-{exposure}.fits',
                  'guider-{exposure}.fits.fz')


class DTSPipeline(object):
    """Simple object for generating pipeline commands.

    Parameters
    ----------
    host : str
        Run the pipeline on this NERSC system.
    ssh : str, optional
        SSH command to use.
    queue : str, optional
        NERSC queue to use.
    nodes : int, optional
        Value for the ``--nersc_maxnodes`` option.
    """
    desi_night = os.path.realpath(os.path.join(os.environ['HOME'],
                                               'bin',
                                               'wrap_desi_night.sh'))

    def __init__(self, host, ssh='ssh', queue='realtime', nodes=25):
        self.host = host
        self.ssh = ssh
        self.queue = queue
        self.nodes = str(nodes)
        return

    def command(night, exposure, command='update'):
        """Generate a ``desi_night`` command to pass to the pipeline.

        Parameters
        ----------
        night : str
            Night of observation.
        exposure : str
            Exposure number.
        command : str, optional
            Specific command to pass to ``desi_night``.

        Returns
        -------
        list
            A command suitable for passing to :class:`subprocess.Popen`.
        """
        cmd = command
        if command == 'science':
            cmd = 'redshifts'
        c = [self.ssh, '-q', self.host,
             self.desi_night, cmd,
             '--night', night,
             '--expid', exposure,
             '--nersc', self.host,
             '--nersc_queue', self.queue,
             '--nersc_maxnodes', self.nodes]
        log.debug(' '.join(c))
        return cmd


class DTSStatus(object):
    """Simple object for interacting with DTS status reports.

    Parameters
    ----------
    directory : str
        Retrieve and store JSON-encoded transfer status data in `directory`.
    """

    def __init__(self, directory):
        self.directory = directory
        self.json = os.path.join(self.directory, 'dts_status.json')
        self.status = list()
        if not os.path.exists(self.directory):
            log.debug("os.makedirs('%s')", self.directory)
            os.makedirs(self.directory)
            for ext in ('html', 'js'):
                src = resource_filename('desispec', 'data/dts/dts_status.' + ext)
                if ext == 'html':
                    shutil.copyfile(src, os.path.join(self.directory, 'index.html'))
                else:
                    shutil.copy(src, self.directory)
            return
        try:
            with open(self.json) as j:
                self.status = json.load(j)
        except FileNotFoundError:
            pass
        return

    def update(self, night, exposure, stage, failure=False, last=None):
        """Update the transfer status.

        Parameters
        ----------
        night : str
            Night of observation.
        exposure : str
            Exposure number.
        stage : str
            Stage of data transfer ('rsync', 'checksum', 'backup', ...).
        failure : bool, optional
            Indicate failure.
        last : str, optional
            Mark this exposure as the last of a given type for the night
            ('arcs', 'flats', 'science').
        """
        k = lambda x: x[0]*10000000 + x[1]
        if last is None:
            l = ''
        else:
            l = last
        ts = int(time.time() * 1000)  # Convert to milliseconds for JS.
        in = int(night)
        if exposure == 'all':
            rows = [[r[0], r[1], stage, not failure, l, ts]
                    for r in self.status if r[0] == in]
        else:
            rows = [[in, int(exposure), stage, not failure, l, ts],]
        for row in rows:
            self.status.insert(0, row)
        self.status = sorted(self.status, key=k, reverse=True)
        with open(self.json, 'w') as j:
            json.dump(self.status, j, indent=None, separators=(',', ':'))


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
    prsr.add_argument('-s', '--sleep', metavar='M', type=int, default=10,
                      help='Sleep M minutes before checking for new data (default %(default)s minutes).')
    prsr.add_argument('-S', '--shadow', action='store_true',
                      help='Observe the actions of another data transfer script but do not make any changes.')
    if len(args) > 0:
        options = prsr.parse_args(args)
    else:  # pragma: no cover
        options = prsr.parse_args()
    return options


def _popen(command):
    """Simple wrapper for :class:`subprocess.Popen` to avoid repeated code.
    """
    log.debug(' '.join(command))
    p = sub.Popen(command, stdout=sub.PIPE, stderr=sub.PIPE)
    out, err = p.communicate()
    return (str(p.returncode), out.decode('utf-8'), err.decode('utf-8'))


def _configure_log(debug, size=100000000, backups=100):
    """Re-configure the default logger returned by ``desiutil.log``.

    Parameters
    ----------
    debug : bool
        If ``True`` set the log level to ``DEBUG``.
    size : int, optional
        Rotate log file after N bytes.
    backups : int, optional
        Keep N old log files.
    """
    global log
    log_filename = os.path.realpath(os.path.join(os.environ['DESI_ROOT'],
                                                 'spectro', 'staging', 'logs',
                                                 'desi_dts.log'))
    log = get_logger(timestamp=True)
    h = log.parent.handlers[0]
    handler = RotatingFileHandler(log_filename, maxBytes=size,
                                  backupCount=backups)
    handler.setFormatter(h.formatter)
    log.parent.removeHandler(h)
    log.parent.addHandler(handler)
    if debug:
        log.setLevel(logging.DEBUG)
    email_from = os.environ['USER'] + '@' + getfqdn()
    email_to = ['desi-data@desi.lbl.gov',]
    handler2 = SMTPHandler('localhost', email_from, email_to,
                           'Critical error reported by desi_dts!')
    formatter2 = logging.Formatter('At %(asctime)s, desi_dts failed with this message:\n\n%(message)s\n\nKia ora koutou,\nThe DESI Collaboration Account',
                                   '%Y-%m-%dT%H:%M:%S %Z')
    handler2.setFormatter(formatter2)
    handler2.setLevel(logging.CRITICAL)
    log.parent.addHandler(handler2)


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
        An integer that indicates the number of checksum mismatches.  A
        value of -1 indicates that the lines in the checksum file does not
        match the number of files in the exposure.
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


def main():
    """Entry point for :command:`desi_dts`.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    options = _options()
    _configure_log(options.debug)
    ssh = 'ssh'
    pipeline = DTSPipeline(options.nersc, ssh=ssh)
    while True:
        log.info('Starting transfer loop.')
        if os.path.exists(options.kill):
            log.info("%s detected, shutting down transfer daemon.", options.kill)
            return 0
        #
        # Find symlinks at KPNO.
        #
        for d in _config():
            status = DTSStatus(os.path.join(os.path.dirname(d.staging), 'status'))
            cmd = [ssh, '-q', 'dts', '/bin/find', d.source, '-type', 'l']
            _, out, err = _popen(cmd)
            links = sorted([x for x in out.split('\n') if x])
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
                        if not options.shadow:
                            os.makedirs(n, exist_ok=True)
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
                        if options.shadow:
                            log.debug(' '.join(cmd))
                            rsync_status = '0'
                        else:
                            rsync_status, out, err = _popen(cmd)
                    else:
                        log.info('%s already transferred.', se)
                        rsync_status = 'done'
                    #
                    # Transfer complete.
                    #
                    if rsync_status == '0':
                        status.update(night, exposure, 'rsync')
                        #
                        # Check permissions.
                        #
                        log.debug("os.chmod('%s', 0o%o)", se, dir_perm)
                        if not options.shadow:
                            os.chmod(se, dir_perm)
                        exposure_files = os.listdir(se)
                        for f in exposure_files:
                            ff = os.path.join(se, f)
                            if os.path.isfile(ff):
                                log.debug("os.chmod('%s', 0o%o)", ff, file_perm)
                                if not options.shadow:
                                    os.chmod(ff, file_perm)
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
                            status.update(night, exposure, 'checksum')
                            #
                            # Set up DESI_SPECTRO_DATA.
                            #
                            dn = os.path.join(d.destination, night)
                            if not os.path.isdir(dn):
                                log.debug("os.makedirs('%s', exist_ok=True)", dn)
                                if not options.shadow:
                                    os.makedirs(dn, exist_ok=True)
                            #
                            # Move data into DESI_SPECTRO_DATA.
                            #
                            if not os.path.isdir(de):
                                log.debug("shutil.move('%s', '%s')", se, dn)
                                if not options.shadow:
                                    shutil.move(se, dn)
                            #
                            # Is this a "realistic" exposure?
                            #
                            if options.pipeline and check_exposure(de, exposure):
                                #
                                # Run update
                                #
                                cmd = pipeline.command(night, exposure)
                                if not options.shadow:
                                    _, out, err = _popen(cmd)
                                done = False
                                for k in ('flats', 'arcs', 'science'):
                                    if os.path.exists(os.path.join(de, '{0}-{1}-{2}.done'.format(k, night, exposure))):
                                        cmd = pipeline.command(night, exposure, command=k)
                                        if not options.shadow:
                                            _, out, err = _popen(cmd)
                                        status.update(night, exposure, 'pipeline', last=k)
                                        done = True
                                if not done:
                                    status.update(night, exposure, 'pipeline')
                            else:
                                log.info("%s/%s appears to be test data. Skipping pipeline activation.", night, exposure)
                        else:
                            log.error("Checksum problem detected for %s/%s!", night, exposure)
                            status.update(night, exposure, 'checksum', failure=True)
                    elif rsync_status == 'done':
                        #
                        # Do nothing, successfully.
                        #
                        pass
                    else:
                        log.error('rsync problem detected!')
                        status.update(night, exposure, 'rsync', failure=True)
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
            if options.shadow:
                ls_file = ls_file.replace('.txt', '.shadow.txt')
            if now >= options.backup:
                if os.path.isdir(os.path.join(d.destination, yesterday)):
                    log.debug("os.remove('%s')", ls_file)
                    os.remove(ls_file)
                    cmd = ['/usr/common/mss/bin/hsi', '-O', ls_file,
                           'ls', '-l', d.hpss]
                    _, out, err = _popen(cmd)
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
                        start_dir = os.getcwd()
                        log.debug("os.chdir('%s')", d.destination)
                        os.chdir(d.destination)
                        cmd = ['/usr/common/mss/bin/htar',
                               '-cvhf', os.path.join(d.hpss, backup_file),
                               '-H', 'crc:verify=all',
                               yesterday]
                        if options.shadow:
                            log.debug(' '.join(cmd))
                        else:
                            _, out, err = _popen(cmd)
                        log.debug("os.chdir('%s')", start_dir)
                        os.chdir(start_dir)
                        status.update(night, 'all', 'backup')
                else:
                    log.warning("No data from %s detected, skipping HPSS backup.", yesterday)
        time.sleep(options.sleep*60)
    return 0

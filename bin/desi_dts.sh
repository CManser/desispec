#!/bin/bash
#
# Configuration
#
# log=${HOME}/desi_dts.log
log=${DESI_ROOT}/spectro/staging/logs/desi_dts.log
[[ -f ${log} ]] || /bin/touch ${log}
#
# Source, staging, destination and hpss should be in 1-1-1-1 correspondence.
#
source_directories=(/data/dts/exposures/raw)
# source_directories=(/data/dts/exposures/test)
staging_directories=($(/bin/realpath ${DESI_ROOT}/spectro/staging/raw))
# staging_directories=($(/bin/realpath ${CSCRATCH}/desi/spectro/staging/raw))
destination_directories=($(/bin/realpath ${DESI_SPECTRO_DATA}))
# destination_directories=($(/bin/realpath ${CSCRATCH}/desi/spectro/data))
hpss_directories=(desi/spectro/data)
n_source=${#source_directories[@]}
# Enable activation of the DESI pipeline.  If this is /bin/false, only
# transfer files.
run_pipeline=/bin/false
pipeline_host=cori
# The existence of this file will shut down data transfers.
kill_switch=${HOME}/stop_dts
# Call this executable on the pipeline host.
# Make sure the real path is actually valid on the pipeline host.
desi_night=$(/bin/realpath ${HOME}/bin/wrap_desi_night.sh)
ssh="/bin/ssh -q ${pipeline_host}"
# Wait this long before checking for new data.
sleep=10m
# sleep=1m
# UTC time in hours to look for delayed files.
ketchup_time=14
# UTC time in hours to trigger HPSS backups.
backup_time=20
#
# Functions
#
function log {
    local l=$(tr '[a-z]' '[A-Z]' <<<$1)
    echo "${l}: $2" >> ${log}
}
function debug {
    log ${FUNCNAME} "$*"
}
function info {
    log ${FUNCNAME} "$*"
}
function warning {
    log ${FUNCNAME} "$*"
}
function error {
    log ${FUNCNAME} "$*"
}
function sprun {
    debug "$*"
    $* >> ${log} 2>&1
    return $?
}
#
# Endless loop!
#
while /bin/true; do
    info $(/bin/date +'%Y-%m-%dT%H:%M:%S%z')
    if [[ -f ${kill_switch} ]]; then
        info "${kill_switch} detected, shutting down transfer daemon."
        exit 0
    fi
    #
    # Find symlinks at KPNO.
    #
    for (( k=0; k < ${n_source}; k++ )); do
        src=${source_directories[$k]}
        staging=${staging_directories[$k]}
        dest=${destination_directories[$k]}
        status_dir=$(/bin/dirname ${staging})/status
        links=$(/bin/ssh -q dts /bin/find ${src} -type l 2>/dev/null | sort)
        if [[ -n "${links}" ]]; then
            for l in ${links}; do
                exposure=$(/bin/basename ${l})
                night=$(/bin/basename $(/bin/dirname ${l}))
                #
                # New night detected?
                #
                [[ ! -d ${staging}/${night} ]] && \
                    sprun /bin/mkdir -p ${staging}/${night} && \
                    sprun /bin/chmod 2750 ${staging}/${night}
                #
                # Has exposure already been transferred?
                #
                if [[ ! -d ${staging}/${night}/${exposure} && \
                    ! -d ${dest}/${night}/${exposure} ]]; then
                    sprun /bin/rsync --verbose --no-motd \
                        --recursive --copy-dirlinks --times --omit-dir-times \
                        dts:${src}/${night}/${exposure}/ ${staging}/${night}/${exposure}/
                    status=$?
                else
                    info "${staging}/${night}/${exposure} already transferred."
                    status="done"
                fi
                #
                # Transfer complete.
                #
                if [[ "${status}" == "0" ]]; then
                    #
                    # Check permissions.
                    #
                    sprun /usr/bin/find ${staging}/${night}/${exposure} -type d -exec chmod 2750 \{\} \;
                    sprun /usr/bin/find ${staging}/${night}/${exposure} -type f -exec chmod 0440 \{\} \;
                    #
                    # Verify checksums.
                    #
                    if [[ -f ${staging}/${night}/${exposure}/checksum-${night}-${exposure}.sha256sum ]]; then
                        (cd ${staging}/${night}/${exposure} && /bin/sha256sum --quiet --check checksum-${night}-${exposure}.sha256sum) &>> ${log}
                        checksum_status=$?
                    else
                        warning "No checksum file for ${night}/${exposure}."
                        checksum_status=0
                    fi
                    #
                    # Did we pass checksums?
                    #
                    if [[ "${checksum_status}" == "0" ]]; then
                        #
                        # Set up DESI_SPECTRO_DATA.
                        #
                        [[ ! -d ${dest}/${night} ]] && \
                            sprun /bin/mkdir -p ${dest}/${night}
                        #
                        # Move data into DESI_SPECTRO_DATA.
                        #
                        [[ ! -d ${dest}/${night}/${exposure} ]] && \
                            sprun /bin/mv ${staging}/${night}/${exposure} ${dest}/${night}
                        #
                        # Is this a "realistic" exposure?
                        #
                        if ${run_pipeline} && \
                            [[ -f ${dest}/${night}/${exposure}/desi-${exposure}.fits.fz && \
                               -f ${dest}/${night}/${exposure}/fibermap-${exposure}.fits ]]; then
                            #
                            # Run update
                            #
                            sprun ${ssh} ${desi_night} update \
                                --night ${night} --expid ${exposure} \
                                --nersc ${pipeline_host} --nersc_queue realtime \
                                --nersc_maxnodes 25
                            #
                            # if (flat|arc) done, run flat|arc update.
                            #
                            if [[ -f ${dest}/${night}/${exposure}/flats-${night}-${exposure}.done ]]; then
                                sprun ${ssh} ${desi_night} flats \
                                    --night ${night} \
                                    --nersc ${pipeline_host} --nersc_queue realtime \
                                    --nersc_maxnodes 25
                                sprun desi_dts_status --directory ${status_dir} --last flats ${night} ${exposure}
                            elif [[ -f ${dest}/${night}/${exposure}/arcs-${night}-${exposure}.done ]]; then
                                sprun ${ssh} ${desi_night} arcs \
                                    --night ${night} \
                                    --nersc ${pipeline_host} --nersc_queue realtime \
                                    --nersc_maxnodes 25
                                sprun desi_dts_status --directory ${status_dir} --last arcs ${night} ${exposure}
                            #
                            # if night done run redshifts
                            #
                            elif [[ -f ${dest}/${night}/${exposure}/science-${night}-${exposure}.done ]]; then
                                sprun ${ssh} ${desi_night} redshifts \
                                    --night ${night} \
                                    --nersc ${pipeline_host} --nersc_queue realtime \
                                    --nersc_maxnodes 25
                                sprun desi_dts_status --directory ${status_dir} --last science ${night} ${exposure}
                            else
                                sprun desi_dts_status --directory ${status_dir} ${night} ${exposure}
                            fi
                        else
                            info "${night}/${exposure} appears to be test data.  Skipping pipeline activation."
                        fi
                    else
                        error "Checksum problem detected for ${night}/${exposure}!"
                        ${run_pipeline} && sprun desi_dts_status --directory ${status_dir} --failure ${night} ${exposure}
                    fi
                elif [[ "${status}" == "done" ]]; then
                    #
                    # Do nothing, successfully.
                    #
                    :
                else
                    error "rsync problem detected for ${night}/${exposure}!"
                    ${run_pipeline} && sprun desi_dts_status --directory ${status_dir} --failure ${night} ${exposure}
                fi
            done
        else
            warning "No links found, check connection."
        fi
        #
        # WARNING: some of the auxilliary files below were created under
        # the assumption that only one source directory exists at KPNO and
        # only one destination directory exists at NERSC.  This should be
        # fixed now, but watch out for this.
        #
        # Do a "catch-up" transfer to catch delayed files in the morning,
        # rather than at noon.
        # 07:00 MST = 14:00 UTC.
        # This script can do nothing about exposures that were never linked
        # into the DTS area at KPNO in the first place.
        #
        yesterday=$(/bin/date --date="@$(($(/bin/date +%s) - 86400))" +'%Y%m%d')
        now=$(/bin/date -u +'%H')
        ketchup_file=$(echo ${dest} | tr '/' '_')
        sync_file=${CSCRATCH}/ketchup${ketchup_file}_${yesterday}.log
        if (( now >= ketchup_time )); then
            if [[ -d ${dest}/${yesterday} ]]; then
                if [[ -f ${sync_file} ]]; then
                    echo "DEBUG: ${sync_file} detected, catch-up transfer is done."
                else
                    /bin/rsync --verbose --no-motd --dry-run \
                        --recursive --copy-dirlinks --times --omit-dir-times \
                        dts:${src}/${yesterday}/ ${dest}/${yesterday}/ &> ${sync_file}
                    changed=$(/usr/bin/grep -E -v '^(receiving|sent|total)' ${sync_file} | \
                        /usr/bin/grep -E -v '^$' | /usr/bin/wc -l)
                    if [[ ${changed} == 0 ]]; then
                        info "No files appear to have changed in ${yesterday}."
                    else
                        warning "New files detected in ${yesterday}!"
                        sprun /usr/bin/cat ${sync_file}
                        sprun /usr/bin/find ${dest}/${yesterday} -type f -exec chmod 0640 \{\} \;
                        sprun /bin/rsync --verbose --no-motd \
                            --recursive --copy-dirlinks --times --omit-dir-times \
                            dts:${src}/${yesterday}/ ${dest}/${yesterday}/
                        sprun /usr/bin/find ${dest}/${yesterday} -type f -exec chmod 0440 \{\} \;
                    fi
                fi
            else
                warning "No data from ${yesterday} detected, skipping catch-up transfer."
            fi
        fi
        #
        # Are any nights eligible for backup?
        # 12:00 MST = 19:00 UTC.
        # Plus one hour just to be safe, so after 20:00 UTC.
        #
        hpss_file=$(echo ${hpss_directories[$k]} | tr '/' '_')
        ls_file=${CSCRATCH}/${hpss_file}.txt
        if (( now >= backup_time )); then
            if [[ -d ${dest}/${yesterday} ]]; then
                #
                # See what's on HPSS.
                #
                sprun /bin/rm -f ${ls_file}
                sprun /usr/common/mss/bin/hsi -O ${ls_file} ls -l ${hpss_directories[$k]}
                #
                # Both a .tar and a .tar.idx file should be present.
                #
                if [[ $(/usr/bin/grep ${yesterday} ${ls_file} | /usr/bin/wc -l) != 2 ]]; then
                    #
                    # Run a final sync of the night and see if anything changed.
                    # This isn't supposed to be necessary, but during
                    # commissioning, all kinds of crazy stuff might happen.
                    #
                    sync_file=${CSCRATCH}/final_sync${ketchup_file}_${yesterday}.log
                    sprun /bin/rm -f ${sync_file}
                    /bin/rsync --verbose --no-motd --dry-run \
                        --recursive --copy-dirlinks --times --omit-dir-times \
                        dts:${src}/${yesterday}/ ${dest}/${yesterday}/ &> ${sync_file}
                    changed=$(/usr/bin/grep -E -v '^(receiving|sent|total)' ${sync_file} | \
                        /usr/bin/grep -E -v '^$' | /usr/bin/wc -l)
                    if [[ ${changed} == 0 ]]; then
                        info "No files appear to have changed in ${yesterday}."
                        sprun /bin/rm -f ${sync_file}
                    else
                        warning "New files detected in ${yesterday}!"
                        sprun /usr/bin/cat ${sync_file}
                        sprun /usr/bin/find ${dest}/${yesterday} -type f -exec chmod 0640 \{\} \;
                        sprun /bin/rsync --verbose --no-motd \
                            --recursive --copy-dirlinks --times --omit-dir-times \
                            dts:${src}/${yesterday}/ ${dest}/${yesterday}/
                        sprun /usr/bin/find ${dest}/${yesterday} -type f -exec chmod 0440 \{\} \;
                    fi
                    #
                    # Issue HTAR command.
                    #
                    (cd ${dest} && \
                        /usr/common/mss/bin/htar -cvhf \
                            ${hpss_directories[$k]}/${hpss_file}_${yesterday}.tar \
                            -H crc:verify=all ${yesterday}) &>> ${log}
                fi
            else
                warning "No data from ${yesterday} detected, skipping HPSS backup."
            fi
        fi
    done
    sprun /bin/sleep ${sleep}
done

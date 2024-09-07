#!/bin/bash
# Lab = {CLVR, ILIAD, IPRL, RAD, REAL, WEIRD, AUTOLab, GuptaLab, IRIS, PennPAL, RAIL, RPL, TRI}
# base_folder = '/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/1.0.1/ILIAD/success'
while true
do
    echo
    echo "#######################################################"
    echo
    srun -A nvr_srl_simpler \
        --partition interactive,polar,polar2,polar3,polar4,grizzly \
        --gpus 8 \
        --time=4:00:00 \
        --pty bash -c "python launch_droid_test.py --base_folder /lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/1.0.1/TRI/success"
    sleep 1m
done
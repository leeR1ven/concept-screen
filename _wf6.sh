#!/bin/bash
cd /root/ana || exit 1
export OMP_NUM_THREADS=1
export ANA_TORQUE=1
export ANA_TORQUE_NM=25
setsid nohup env ANA_GO2_XML=/root/go2models/scene.xml ANA_TORQUE=1 ANA_TORQUE_NM=25 \
  python3 experiments/p49_walk_far.py --gens 60 --pop 40 --jobs 30 \
  > walkfar.log 2>&1 < /dev/null &
disown
sleep 150
echo PROCS=$(pgrep -fc p49_walk_far)
tail -4 walkfar.log

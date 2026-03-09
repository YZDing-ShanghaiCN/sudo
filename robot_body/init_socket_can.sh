#!/bin/bash
BITRATE=1000000      # 波特率变量，方便修改
TXQLEN=65535         # 发送队列长度
CAN_COUNT=${1:-5}    # CAN 设备数量（默认can0 ~ can3）
for ((i=0; i<$CAN_COUNT; i++)); do
    CAN_IF="can$i"
    echo "配置 $CAN_IF, 波特率=$BITRATE, 队列长度=$TXQLEN"

    sudo ip link set down $CAN_IF
    sudo ip link set $CAN_IF type can bitrate $BITRATE
    sudo ip link set up $CAN_IF
    sudo ifconfig $CAN_IF txqueuelen $TXQLEN
done
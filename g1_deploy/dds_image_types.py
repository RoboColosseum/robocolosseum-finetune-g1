"""ROS 2 sensor_msgs/msg/CompressedImage wire types (XCDR1)."""
from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.annotations import final
from cyclonedds.idl.types import int32, uint32, uint8, sequence


@dataclass
@final
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32
    nanosec: uint32


@dataclass
@final
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time
    frame_id: str


@dataclass
@final
class CompressedImage(IdlStruct, typename="sensor_msgs::msg::dds_::CompressedImage_"):
    header: Header
    format: str
    data: sequence[uint8]

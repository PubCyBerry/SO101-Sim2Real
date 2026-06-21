"""SO-101 EEPROM/calibration readback을 mutation 없이 검증한다."""

from __future__ import annotations

from typing import Any, Mapping


_GLOBAL_REGISTERS = {
    "Operating_Mode": "operating_mode",
    "P_Coefficient": "p_coefficient",
    "I_Coefficient": "i_coefficient",
    "D_Coefficient": "d_coefficient",
    "Return_Delay_Time": "return_delay_time",
    "Acceleration": "acceleration",
    "Torque_Enable": "required_preflight_torque_enable",
}

_JOINT_REGISTERS = {
    "Homing_Offset": "homing_offset",
    "Min_Position_Limit": "range_min",
    "Max_Position_Limit": "range_max",
}

_GRIPPER_REGISTERS = {
    "Max_Torque_Limit": "max_torque_limit",
    "Protection_Current": "protection_current",
    "Overload_Torque": "overload_torque",
}


def verify_motor_profile(bus, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Torque-off bus의 EEPROM/readback과 LeRobot calibration snapshot을 비교한다."""

    mismatches: list[str] = []
    registers: dict[str, dict[str, int]] = {}
    for register, key in _GLOBAL_REGISTERS.items():
        values = {
            name: int(value)
            for name, value in bus.sync_read(register, normalize=False).items()
        }
        registers[register] = values
        expected_value = int(expected[key])
        for name, value in values.items():
            if value != expected_value:
                mismatches.append(
                    f"{name}.{register}: actual={value}, expected={expected_value}"
                )

    joints = expected["joints"]
    for register, key in _JOINT_REGISTERS.items():
        values = {
            name: int(value)
            for name, value in bus.sync_read(register, normalize=False).items()
        }
        registers[register] = values
        for name, value in values.items():
            expected_value = int(joints[name][key])
            if value != expected_value:
                mismatches.append(
                    f"{name}.{register}: actual={value}, expected={expected_value}"
                )

    gripper_expected = expected["gripper"]
    for register, key in _GRIPPER_REGISTERS.items():
        value = int(bus.read(register, "gripper", normalize=False))
        registers[register] = {"gripper": value}
        expected_value = int(gripper_expected[key])
        if value != expected_value:
            mismatches.append(
                f"gripper.{register}: actual={value}, expected={expected_value}"
            )

    calibration = {}
    for name, item in bus.calibration.items():
        actual = {
            "id": int(item.id),
            "drive_mode": int(item.drive_mode),
            "homing_offset": int(item.homing_offset),
            "range_min": int(item.range_min),
            "range_max": int(item.range_max),
        }
        calibration[name] = actual
        for key, expected_value in joints[name].items():
            if int(actual[key]) != int(expected_value):
                mismatches.append(
                    f"{name}.calibration.{key}: actual={actual[key]}, expected={expected_value}"
                )

    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "registers": registers,
        "calibration": calibration,
    }

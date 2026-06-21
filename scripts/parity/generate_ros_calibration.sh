#!/usr/bin/env bash
set -euo pipefail

input="${1:-calibration/so101_canonical.json}"
output="${2:-ros2_ws/src/so101_bringup/config/hardware/follower_joints.yaml}"

tmp_json="$(mktemp)"
trap 'rm -f "$tmp_json"' EXIT

jq '
  .motor_profile.expected as $profile
  | {
      joints: (
        $profile.joints
        | with_entries(
            .value += {
              p_coefficient: $profile.p_coefficient,
              i_coefficient: $profile.i_coefficient,
              d_coefficient: $profile.d_coefficient,
              return_delay_time: $profile.return_delay_time,
              acceleration: $profile.acceleration
            }
          )
        | .gripper += $profile.gripper
      )
    }
' "$input" > "$tmp_json"

mkdir -p "$(dirname "$output")"
yq -P -o=yaml '.' "$tmp_json" > "$output"

expected="$(jq -S -c '.' "$tmp_json")"
actual="$(yq -o=json '.' "$output" | jq -S -c '.')"
if [[ "$expected" != "$actual" ]]; then
  echo "ROS YAML round-trip mismatch" >&2
  exit 1
fi

echo "$output"

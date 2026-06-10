#pragma once

#include <feetech_driver/communication_protocol.hpp>
#include <feetech_driver/serial_port.hpp>
#include <hardware_interface/handle.hpp>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/system_interface.hpp>
#include <map>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <vector>

#if __has_include(<hardware_interface/hardware_interface/version.h>)
#include <hardware_interface/hardware_interface/version.h>
#else
#include <hardware_interface/version.h>
#endif

#include "feetech_ros2_driver/joint_config.hpp"

namespace feetech_ros2_driver {

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class FeetechHardwareInterface : public hardware_interface::SystemInterface {
 public:
#if HARDWARE_INTERFACE_VERSION_GTE(4, 34, 0)
  CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams& params) override;
#else
  CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override;
#endif

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  hardware_interface::return_type write(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  std::unique_ptr<feetech_driver::CommunicationProtocol> communication_protocol_;

  std::vector<double> hw_positions_;
  std::vector<double> state_hw_positions_;
  std::vector<double> state_hw_velocities_;
  std::vector<uint8_t> previous_hw_positions_;

  std::vector<uint8_t> joint_ids_;

  // USB-IP(WSL2 mirrored) 지연 스파이크로 인한 일시적 read timeout 톨러런스:
  // 단일 실패에 hardware 를 deactivate 하지 않고, 연속 실패가 임계치를 넘을 때만 ERROR 를 반환한다.
  // (실패 cycle 은 마지막으로 읽은 상태를 그대로 유지한 채 skip 한다.)
  std::size_t consecutive_read_failures_ = 0;
  static constexpr std::size_t kMaxConsecutiveReadFailures = 10;

  CallbackReturn init_transport_();
  CallbackReturn load_yaml_config_and_warn_(JointIdConfigMap& out_yaml);
  CallbackReturn configure_joints_(const JointIdConfigMap& yaml_by_id);
  CallbackReturn validate_model_series_();
};
}  // namespace feetech_ros2_driver

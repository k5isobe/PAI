#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <cstdio>
#include <array>
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("pick_and_place");

double to_radians(const double deg_angle) {
  return deg_angle * M_PI / 180.0;
}

// シェル引数の安全なエスケープ
std::string shell_quote(const std::string& value) {
    std::string result = "'";
    for (char c : value) {
        if (c == '\'') {
            result += "'\\''";
        } else {
            result += c;
        }
    }
    result += "'";
    return result;
}

// LLMで自然言語指示を解釈
bool interpret_instruction(
    const std::string& instruction,
    double& place_x,
    double& place_y) {
    const std::string script =
        "/home/ubuntu/airobot_ws/src/crane_plus/"
        "crane_plus_examples/scripts/interpret_instruction.py";

    const std::string command =
        "python3 " + shell_quote(script) +
        " " + shell_quote(instruction) + " 2>&1";

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return false;
    }

    char buffer[256];
    std::string output;

    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    int status = pclose(pipe);

    if (status != 0) {
        RCLCPP_ERROR(LOGGER, "LLM呼び出し失敗: %s", output.c_str());
        return false;
    }

    if (sscanf(output.c_str(), "%lf %lf",
               &place_x, &place_y) != 2) {
        RCLCPP_ERROR(LOGGER, "LLM出力の解析に失敗: %s", output.c_str());
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto move_group_arm_node = rclcpp::Node::make_shared("move_group_arm_node", node_options);
  auto move_group_gripper_node = rclcpp::Node::make_shared("move_group_gripper_node", node_options);

  // カメラからのピクセル座標を受け取るサブスクライバ
  double detected_u = 0.0;
  double detected_v = 0.0;
  bool detection_received = false;
  auto pixel_sub = move_group_arm_node->create_subscription<
    geometry_msgs::msg::PointStamped>(
    "/detected_box/pixel", 10,
    [&](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
      detected_u = msg->point.x;
      detected_v = msg->point.y;
      detection_received = true;
    });

  // For current state monitor (executor thread)
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(move_group_arm_node);
  executor.add_node(move_group_gripper_node);
  std::thread([&executor]() {executor.spin();}).detach();

  MoveGroupInterface move_group_arm(move_group_arm_node, "arm");
  move_group_arm.setMaxVelocityScalingFactor(1.0);  // Set 0.0 ~ 1.0
  move_group_arm.setMaxAccelerationScalingFactor(1.0);  // Set 0.0 ~ 1.0

  MoveGroupInterface move_group_gripper(move_group_gripper_node, "gripper");
  move_group_gripper.setMaxVelocityScalingFactor(1.0);  // Set 0.0 ~ 1.0
  move_group_gripper.setMaxAccelerationScalingFactor(1.0);  // Set 0.0 ~ 1.0
  auto gripper_joint_values = move_group_gripper.getCurrentJointValues();
  double GRIPPER_DEFAULT = 0.0;
  double GRIPPER_OPEN = to_radians(-30);
  double GRIPPER_CLOSE = to_radians(10);

  move_group_arm.setNamedTarget("vertical");
  move_group_arm.move();

  gripper_joint_values[0] = GRIPPER_DEFAULT;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();

  // ----- Picking Preparation -----
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();

  gripper_joint_values[0] = GRIPPER_OPEN;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();

  // カメラ画像の受信を待機
  for (int i = 0; i < 20 && !detection_received; ++i) {
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }

  // ----- 【ステップ2】実測データに基づくアフィン変換による座標算出 -----
  double pickup_x = 0.0;
  double pickup_y = -0.09;

  if (detection_received) {
    // 5点の実測データから求めたアフィン変換係数
    // x = a * u + b * v + c
    // y = d * u + e * v + f
    const double a = -0.000482;
    const double b =  0.000171;
    const double c =  0.125400;

    const double d =  0.000543;
    const double e =  0.000752;
    const double f = -0.451200;

    pickup_x = a * detected_u + b * detected_v + c;
    pickup_y = d * detected_u + e * detected_v + f;

    RCLCPP_INFO(LOGGER, "【アフィン変換】検出 u=%.1f, v=%.1f -> 計算 pickup_x=%.3f, pickup_y=%.3f",
                detected_u, detected_v, pickup_x, pickup_y);
  } else {
    RCLCPP_WARN(LOGGER, "【警告】カメラ検出なし。デフォルトの安全座標を使用します。");
  }

  RCLCPP_INFO(LOGGER, "000");
  
  // 安全範囲のガード（pickup_xがマイナスにならないように厳しく制限、および範囲外チェック）
  if (pickup_x < 0.0 || pickup_x > 0.15 || pickup_y < -0.17 || pickup_y > 0.02) {
    pickup_x = 0.0;
    pickup_y = -0.09;
    RCLCPP_WARN(LOGGER, "計算された座標が範囲外（特にxが負など）のため、安全なデフォルト座標にフォールバックします。");
  }

  geometry_msgs::msg::Pose target_pose;
  tf2::Quaternion q;
  target_pose.position.x = pickup_x;
  target_pose.position.y = pickup_y;
  target_pose.position.z = 0.17;
  // test
  /*
  target_pose.position.x = 0.025; // ng
  target_pose.position.x = 0.01; // ng
  target_pose.position.x = 0.001; // ng
  target_pose.position.x = 0.0; // ok
  target_pose.position.y = -0.09; // ok
  target_pose.position.y = -0.2; // ng
  */
  target_pose.position.z = 0.17;
  q.setRPY(to_radians(0), to_radians(90), to_radians(-90));
  target_pose.orientation = tf2::toMsg(q);

  RCLCPP_INFO(LOGGER, "010");
  
  //bool ik_ok = move_group_arm.setJointValueTarget(target_pose);
  bool ik_ok = move_group_arm.setApproximateJointValueTarget(target_pose);

  RCLCPP_INFO(
    LOGGER,
    "IK result: %s",
    ik_ok ? "SUCCESS" : "FAILED");

  if (!ik_ok) {
    RCLCPP_ERROR(LOGGER, "指定した位置・姿勢のIK解が見つかりません。");
    rclcpp::shutdown();
    return 1;
  }

  std::vector<double> ik_joints;
  move_group_arm.getJointValueTarget(ik_joints);

  for (size_t i = 0; i < ik_joints.size(); ++i) {
    RCLCPP_INFO(
      LOGGER,
      "IK joint[%zu] = %.4f rad",
      i,
      ik_joints[i]);
  }

  RCLCPP_INFO(LOGGER, "Joint target planning test");

  moveit::planning_interface::MoveGroupInterface::Plan test_plan;

  auto plan_result = move_group_arm.plan(test_plan);

  RCLCPP_INFO(
    LOGGER,
    "Joint target plan result: %d",
    plan_result.val);

  if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_INFO(LOGGER, "Joint target planning succeeded.");
  } else {
    RCLCPP_ERROR(LOGGER, "Joint target planning failed.");
  }


  move_group_arm.setPoseTarget(target_pose);
  move_group_arm.move();

  RCLCPP_INFO(LOGGER, "020");
  

  q.setRPY(to_radians(0), to_radians(180), to_radians(-90));
  target_pose.orientation = tf2::toMsg(q);
  move_group_arm.setPoseTarget(target_pose);
  move_group_arm.move();

  target_pose.position.z = 0.14;
  move_group_arm.setPoseTarget(target_pose);
  move_group_arm.move();

  RCLCPP_INFO(LOGGER, "040");
  

  // Grasp
  gripper_joint_values[0] = GRIPPER_CLOSE;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();

  target_pose.position.z = 0.17;
  move_group_arm.setPoseTarget(target_pose);
  move_group_arm.move();

  RCLCPP_INFO(LOGGER, "060");
  

  // ----- Natural-language instruction -----
  std::string instruction = "右側に置く";
  move_group_arm_node->get_parameter("instruction", instruction);

  double place_x = 0.15;
  double place_y = 0.0;

  RCLCPP_INFO(LOGGER, "080");

  /*
  if (instruction.find("手前") != std::string::npos) {
    place_x = 0.10;
    place_y = -0.05;
  } else if (instruction.find("右") != std::string::npos) {
    place_x = 0.15;
    place_y = 0.0;
  } else {
    std::cout << "指示を認識できないため、右側に置きます。\n";
  }*/

  // ----- LLMによる自然言語指示の解釈 -----

  RCLCPP_INFO(LOGGER, "LLMへの指示: %s", instruction.c_str());

  if (!interpret_instruction(instruction, place_x, place_y)) {
      RCLCPP_ERROR(LOGGER, "AIによる指示解釈に失敗。動作を中止します。");
      rclcpp::shutdown();
      return 1;
  }

  RCLCPP_INFO(
      LOGGER,
      "AIによる解釈結果: 配置先 x=%.3f, y=%.3f",
      place_x, place_y);

  RCLCPP_INFO(LOGGER, "100");
  
  std::cout << "配置先: x=" << place_x << ", y=" << place_y << std::endl;

  // ----- Placing Preparation -----
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();

  target_pose.position.x = place_x;
  target_pose.position.y = place_y;
  target_pose.position.z = 0.06;
  q.setRPY(to_radians(0), to_radians(90), to_radians(0));
  target_pose.orientation = tf2::toMsg(q);
  move_group_arm.setPoseTarget(target_pose);
  move_group_arm.move();

  // Release
  gripper_joint_values[0] = GRIPPER_OPEN;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();

  // Return to home and vertical pose
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();
  move_group_arm.setNamedTarget("vertical");
  move_group_arm.move();

  gripper_joint_values[0] = GRIPPER_DEFAULT;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();

  rclcpp::shutdown();
  return 0;
}
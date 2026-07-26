// arm_monitor.h - background poller + panel renderer for the SO-101's six
// Feetech STS3215 servos.
//
// The poll thread is read-only (see sts_bus.h) and runs independently of the
// camera loop, so a slow or absent servo bus can never stall rendering.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#include "sts_bus.h"

namespace arm {

constexpr int kNumJoints = 6;

// SO-101 servo IDs are 1..6 from the base out to the gripper.
const std::array<const char*, kNumJoints> kJointNames = {
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper"};

struct Joint {
  bool responding = false;
  sts::Present p{};
};

struct Snapshot {
  bool port_open = false;
  std::string status;  // human-readable port / error state
  std::array<Joint, kNumJoints> joints{};
  double poll_hz = 0.0;
};

// Raw ticks -> degrees about the servo's mid-travel point. This is the servo's
// own frame: no homing offsets or joint-direction calibration are applied, so
// treat it as "where the servo is", not "where the arm model thinks it is".
inline double ticks_to_deg(uint16_t ticks) {
  return (static_cast<double>(ticks) - sts::kTicksPerRev / 2.0) * 360.0 /
         sts::kTicksPerRev;
}

class Monitor {
 public:
  ~Monitor() { stop(); }

  void start(const std::string& port, int baud) {
    stop();
    port_ = port;
    baud_ = baud;
    running_ = true;
    thread_ = std::thread(&Monitor::run, this);
  }

  void stop() {
    running_ = false;
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  Snapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return snap_;
  }

 private:
  void run() {
    sts::Bus bus;
    std::string err;
    auto last = std::chrono::steady_clock::now();

    while (running_) {
      if (!bus.is_open()) {
        if (!bus.open_port(port_, baud_, &err)) {
          {
            std::lock_guard<std::mutex> lock(mu_);
            snap_ = Snapshot{};
            snap_.status = err;
          }
          // Port may appear later (arm plugged in mid-run); retry slowly.
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
      }

      Snapshot next;
      next.port_open = true;
      next.status = port_ + " @ " + std::to_string(baud_ / 1000000) + " Mbaud";
      int responding = 0;
      for (int i = 0; i < kNumJoints; ++i) {
        sts::Present p;
        if (bus.read_present(static_cast<uint8_t>(i + 1), &p)) {
          next.joints[i].responding = true;
          next.joints[i].p = p;
          ++responding;
        }
      }
      if (responding == 0) {
        next.status = port_ + ": no servos responding (arm powered?)";
      }

      const auto now = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(now - last).count();
      last = now;
      {
        std::lock_guard<std::mutex> lock(mu_);
        next.poll_hz = (dt > 0.0) ? (snap_.poll_hz == 0.0 ? 1.0 / dt
                                                          : 0.8 * snap_.poll_hz +
                                                                0.2 / dt)
                                  : snap_.poll_hz;
        snap_ = next;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  mutable std::mutex mu_;
  Snapshot snap_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::string port_;
  int baud_ = 1000000;
};

// Renders the joint panel at the given size. Scales its typography to the
// panel height so it stays legible when the window is blown up to full screen.
inline cv::Mat render_panel(const Snapshot& s, int width, int height) {
  cv::Mat panel(height, width, CV_8UC3, cv::Scalar(24, 20, 18));

  const double k = height / 720.0;  // layout scale factor
  const auto text = [&](const std::string& t, cv::Point at, double size,
                        cv::Scalar color, int thick = 1) {
    cv::putText(panel, t, at, cv::FONT_HERSHEY_SIMPLEX, size * k, color,
                std::max(1, static_cast<int>(thick * k)), cv::LINE_AA);
  };

  text("SO-101  joint positions", {int(24 * k), int(46 * k)}, 0.85,
       {255, 255, 255}, 2);
  text(s.status, {int(24 * k), int(78 * k)}, 0.5,
       s.port_open ? cv::Scalar(170, 200, 170) : cv::Scalar(120, 120, 255));
  if (s.port_open) {
    char hz[48];
    std::snprintf(hz, sizeof(hz), "%.0f Hz poll", s.poll_hz);
    text(hz, {int(width - 150 * k), int(78 * k)}, 0.5, {160, 160, 160});
  }

  const int row_h = static_cast<int>(96 * k);
  const int top = static_cast<int>(112 * k);
  const int x0 = static_cast<int>(24 * k);
  const int bar_w = width - static_cast<int>(48 * k);
  const int bar_h = static_cast<int>(20 * k);

  for (int i = 0; i < kNumJoints; ++i) {
    const int y = top + i * row_h;
    const Joint& j = s.joints[i];

    char label[96];
    std::snprintf(label, sizeof(label), "%d  %s", i + 1, kJointNames[i]);
    text(label, {x0, y + int(22 * k)}, 0.62, {235, 235, 235}, 2);

    if (!j.responding) {
      text("no response", {x0 + int(300 * k), y + int(22 * k)}, 0.62,
           {110, 110, 255}, 2);
      cv::rectangle(panel, {x0, y + int(36 * k)}, {x0 + bar_w, y + int(36 * k) + bar_h},
                    {60, 60, 60}, 1);
      continue;
    }

    char val[96];
    std::snprintf(val, sizeof(val), "%+7.1f deg   %4u tk", ticks_to_deg(j.p.position),
                  j.p.position);
    const int vw = static_cast<int>(330 * k);
    text(val, {x0 + bar_w - vw, y + int(22 * k)}, 0.62, {120, 240, 255}, 2);

    // Position bar across the servo's full 0..4095 travel, with a mid mark.
    const cv::Point bl(x0, y + int(36 * k));
    const cv::Point br(x0 + bar_w, y + int(36 * k) + bar_h);
    cv::rectangle(panel, bl, br, {70, 70, 70}, 1);
    const int fill =
        static_cast<int>(bar_w * (j.p.position / double(sts::kTicksPerRev - 1)));
    cv::rectangle(panel, bl, {x0 + fill, br.y}, {90, 200, 120}, cv::FILLED);
    const int mid = x0 + bar_w / 2;
    cv::line(panel, {mid, bl.y - int(3 * k)}, {mid, br.y + int(3 * k)},
             {150, 150, 150}, std::max(1, int(k)));

    // Secondary line: load, temperature, supply voltage.
    char aux[128];
    std::snprintf(aux, sizeof(aux), "load %+5.1f%%    %2u C    %.1f V",
                  j.p.load / 10.0, j.p.temperature, j.p.voltage / 10.0);
    const cv::Scalar aux_color = (j.p.temperature >= 55)
                                     ? cv::Scalar(80, 140, 255)   // hot
                                     : cv::Scalar(170, 170, 170);
    text(aux, {x0, y + int(78 * k)}, 0.48, aux_color);
  }

  text("read-only: position/load/temp registers, no servo writes",
       {x0, height - int(18 * k)}, 0.44, {130, 130, 130});
  return panel;
}

}  // namespace arm

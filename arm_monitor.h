// arm_monitor.h - background poller + panel renderer for the SO-101's six
// Feetech STS3215 servos.
//
// The poll thread is read-only (see sts_bus.h) and runs independently of the
// camera loop, so a slow or absent servo bus can never stall rendering.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <glob.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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

// Resolves the servo bus device path. "auto" prefers the /dev/serial/by-id
// symlink for the CH34x bridge on the SO-101 driver board: that name is derived
// from the chip's serial number, so it follows the arm across replugs and USB
// hubs, whereas the ttyACMn index does not.
inline std::vector<std::string> resolve_ports() {
  std::vector<std::string> paths;
  glob_t g{};
  if (::glob("/dev/serial/by-id/*1a86*", GLOB_NOSORT, nullptr, &g) == 0) {
    for (size_t i = 0; i < g.gl_pathc; ++i) {
      paths.emplace_back(g.gl_pathv[i]);
    }
  }
  ::globfree(&g);
  std::sort(paths.begin(), paths.end());
  return paths;
}

inline std::string resolve_port(const std::string& configured) {
  if (configured != "auto") {
    return configured;
  }
  const std::vector<std::string> paths = resolve_ports();
  return paths.empty() ? "/dev/ttyACM0" : paths.front();
}

// The CH343 serial number is the only stable identifier for a driver board, so
// map the known boards to their role. Anything else is labelled by serial.
inline std::string label_for(const std::string& by_id_path) {
  const std::string tag = "USB_Single_Serial_";
  const size_t a = by_id_path.find(tag);
  std::string serial = "unknown";
  if (a != std::string::npos) {
    const size_t b = by_id_path.find("-if", a);
    serial = by_id_path.substr(a + tag.size(),
                               (b == std::string::npos) ? std::string::npos
                                                        : b - a - tag.size());
  }
  if (serial == "5AE6082981") return "follower  (" + serial + ")";
  if (serial == "5AE6085251") return "leader / teleop  (" + serial + ")";
  return "SO-101  (" + serial + ")";
}

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
    std::string path;
    int dead_cycles = 0;
    auto last = std::chrono::steady_clock::now();

    while (running_) {
      if (!bus.is_open()) {
        // Re-resolve every time: an unplug/replug can land the arm on a
        // different ttyACMn, and it may move again behind a hub.
        path = resolve_port(port_);
        if (!bus.open_port(path, baud_, &err)) {
          {
            std::lock_guard<std::mutex> lock(mu_);
            snap_ = Snapshot{};
            snap_.status = err;
          }
          // Port may appear later (arm plugged in mid-run); retry slowly.
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
        dead_cycles = 0;
      }

      Snapshot next;
      next.port_open = true;
      next.status = path + " @ " + std::to_string(baud_ / 1000000) + " Mbaud";
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
        next.status = path + ": no servos responding (arm powered?)";
        // A yanked USB device leaves a stale fd that reads forever without
        // ever recovering, so drop the port and let the next pass re-resolve.
        if (++dead_cycles >= kDeadCyclesBeforeReopen) {
          bus.close_port();
          dead_cycles = 0;
          next.status = "reconnecting...";
        }
      } else {
        dead_cycles = 0;
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

  // ~10 failed polls (about half a second) before we assume the port is gone.
  static constexpr int kDeadCyclesBeforeReopen = 10;

  mutable std::mutex mu_;
  Snapshot snap_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::string port_;
  int baud_ = 1000000;
};

// Renders the joint panel at the given size. Scales its typography to the
// panel height so it stays legible when the window is blown up to full screen.
inline cv::Mat render_panel(const Snapshot& s, int width, int height,
                            const std::string& title) {
  cv::Mat panel(height, width, CV_8UC3, cv::Scalar(24, 20, 18));

  // Two arms share one column, so below a threshold drop the per-servo
  // load/temp/voltage line rather than shrink everything into illegibility.
  const bool compact = height < 480;
  const double k = compact ? height / 420.0 : height / 720.0;
  const auto text = [&](const std::string& t, cv::Point at, double size,
                        cv::Scalar color, int thick = 1) {
    cv::putText(panel, t, at, cv::FONT_HERSHEY_SIMPLEX, size * k, color,
                std::max(1, static_cast<int>(thick * k)), cv::LINE_AA);
  };

  text(title, {int(24 * k), int(46 * k)}, 0.8, {255, 255, 255}, 2);
  text(s.status, {int(24 * k), int(78 * k)}, 0.46,
       s.port_open ? cv::Scalar(170, 200, 170) : cv::Scalar(120, 120, 255));
  if (s.port_open) {
    char hz[48];
    std::snprintf(hz, sizeof(hz), "%.0f Hz poll", s.poll_hz);
    text(hz, {int(width - 150 * k), int(78 * k)}, 0.46, {160, 160, 160});
  }

  // Derive the row pitch from the space actually available rather than a fixed
  // constant, so the last joint can never be clipped off the bottom.
  const int top = static_cast<int>((compact ? 96 : 112) * k);
  const int footer = static_cast<int>((compact ? 6 : 30) * k);
  const int row_h = (height - top - footer) / kNumJoints;
  const int x0 = static_cast<int>(24 * k);
  const int bar_w = width - static_cast<int>(48 * k);
  const int bar_h = static_cast<int>(20 * k);

  for (int i = 0; i < kNumJoints; ++i) {
    const int y = top + i * row_h;
    const Joint& j = s.joints[i];

    // Bar sits just under the text line, within this row's pitch.
    const int text_y = y + static_cast<int>(row_h * 0.35);
    const int bar_y = y + static_cast<int>(row_h * 0.45);

    char label[96];
    std::snprintf(label, sizeof(label), "%d  %s", i + 1, kJointNames[i]);
    text(label, {x0, text_y}, 0.62, {235, 235, 235}, 2);

    if (!j.responding) {
      text("no response", {x0 + int(300 * k), text_y}, 0.62, {110, 110, 255}, 2);
      cv::rectangle(panel, {x0, bar_y}, {x0 + bar_w, bar_y + bar_h}, {60, 60, 60},
                    1);
      continue;
    }

    char val[96];
    std::snprintf(val, sizeof(val), "%+7.1f deg   %4u tk", ticks_to_deg(j.p.position),
                  j.p.position);
    const int vw = static_cast<int>(330 * k);
    text(val, {x0 + bar_w - vw, text_y}, 0.62, {120, 240, 255}, 2);

    // Position bar across the servo's full 0..4095 travel, with a mid mark.
    const cv::Point bl(x0, bar_y);
    const cv::Point br(x0 + bar_w, bar_y + bar_h);
    cv::rectangle(panel, bl, br, {70, 70, 70}, 1);
    const int fill =
        static_cast<int>(bar_w * (j.p.position / double(sts::kTicksPerRev - 1)));
    cv::rectangle(panel, bl, {x0 + fill, br.y}, {90, 200, 120}, cv::FILLED);
    const int mid = x0 + bar_w / 2;
    cv::line(panel, {mid, bl.y - int(3 * k)}, {mid, br.y + int(3 * k)},
             {150, 150, 150}, std::max(1, int(k)));

    if (!compact) {
      // Secondary line: load, temperature, supply voltage.
      char aux[128];
      std::snprintf(aux, sizeof(aux), "load %+5.1f%%    %2u C    %.1f V",
                    j.p.load / 10.0, j.p.temperature, j.p.voltage / 10.0);
      const cv::Scalar aux_color = (j.p.temperature >= 55)
                                       ? cv::Scalar(80, 140, 255)   // hot
                                       : cv::Scalar(170, 170, 170);
      text(aux, {x0, y + static_cast<int>(row_h * 0.82)}, 0.48, aux_color);
    }
  }

  if (!compact) {
    text("read-only: position/load/temp registers, no servo writes",
         {x0, height - int(18 * k)}, 0.44, {130, 130, 130});
  }
  return panel;
}

// Stacks one panel per arm into a single column of the given size.
inline cv::Mat render_panels(const std::vector<std::string>& titles,
                             const std::vector<Snapshot>& snaps, int width,
                             int height) {
  if (snaps.empty()) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(24, 20, 18));
  }
  cv::Mat column;
  const int n = static_cast<int>(snaps.size());
  for (int i = 0; i < n; ++i) {
    // Give the last panel the rounding remainder so the column is exact.
    const int h = (i == n - 1) ? height - (height / n) * (n - 1) : height / n;
    cv::Mat p = render_panel(snaps[i], width, h, titles[i]);
    if (i > 0) {
      cv::line(p, {0, 0}, {width - 1, 0}, {70, 70, 70}, 1);
    }
    if (column.empty()) {
      column = p;
    } else {
      cv::vconcat(column, p, column);
    }
  }
  return column;
}

}  // namespace arm

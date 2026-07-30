// usb_cam.h - background grabber for a V4L2 USB camera (the wrist cam).
//
// Mirrors arm::Monitor: the capture thread owns the device and only ever
// publishes its latest frame, so a slow MJPG decode or an unplugged camera
// can never stall the D455 render loop.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace usbcam {

// Innomaker U20CAM on the wrist. The by-id path survives replugs and hub
// changes; the D455 owns /dev/video0-5, so never fall back to a bare index.
constexpr const char* kDefaultDevice =
    "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0";
constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr double kFps = 30.0;

inline std::string short_name(const std::string& dev) {
  const size_t slash = dev.find_last_of('/');
  return slash == std::string::npos ? dev : dev.substr(slash + 1);
}

struct Snapshot {
  cv::Mat frame;       // latest BGR frame; empty until the first capture
  bool open = false;
  std::string status;  // human-readable device / error state
  double fps = 0.0;    // measured capture rate
};

class Monitor {
 public:
  ~Monitor() { stop(); }

  void start(const std::string& device) {
    stop();
    device_ = device;
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
    cv::VideoCapture cap;
    double fps = 0.0;
    int drops = 0;
    auto last = std::chrono::steady_clock::now();

    while (running_) {
      if (!cap.isOpened()) {
        if (!cap.open(device_, cv::CAP_V4L2)) {
          publish(cv::Mat(), false, short_name(device_) + ": waiting for device",
                  0.0);
          // The camera may appear later (plugged in mid-run); retry slowly.
          for (int i = 0; i < 10 && running_; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          continue;
        }
        // Order matters for V4L2: pixel format first, then geometry, then rate.
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, kWidth);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, kHeight);
        cap.set(cv::CAP_PROP_FPS, kFps);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // prefer latest frame over backlog
        drops = 0;
        fps = 0.0;
      }

      cv::Mat frame;
      if (!cap.read(frame) || frame.empty()) {
        // A yanked camera fails reads forever without recovering (VIDIOC
        // errno 19), so after ~half a second drop the device and re-open.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (++drops >= kDropsBeforeReopen) {
          cap.release();
          publish(cv::Mat(), false, short_name(device_) + ": reconnecting...",
                  0.0);
          drops = 0;
        }
        continue;
      }
      drops = 0;

      const auto now = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(now - last).count();
      last = now;
      if (dt > 0.0) {
        fps = (fps == 0.0) ? 1.0 / dt : 0.9 * fps + 0.1 / dt;
      }

      // A fresh Mat every read: consumers can share the published buffer
      // without ever racing a write.
      publish(std::move(frame), true, short_name(device_), fps);
    }
  }

  void publish(cv::Mat frame, bool open, const std::string& status, double fps) {
    std::lock_guard<std::mutex> lock(mu_);
    snap_.frame = std::move(frame);
    snap_.open = open;
    snap_.status = status;
    snap_.fps = fps;
  }

  // ~100 failed reads (about half a second) before we assume the port is gone.
  static constexpr int kDropsBeforeReopen = 100;

  mutable std::mutex mu_;
  Snapshot snap_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::string device_;
};

}  // namespace usbcam

// d455_viewer.cpp - stream RGB + depth from an Intel RealSense D455 and show
// them side by side in a large window on the desktop.
//
//   q / ESC : quit
//   a       : toggle depth->color alignment
//   c       : cycle depth colormap
//   l       : toggle layout (side-by-side <-> stacked)
//   f       : toggle true fullscreen
//   s       : save a PNG snapshot of the current view
//
// Options: --fullscreen, --scale <0.1..1.0>, --headless
//
// Build: see CMakeLists.txt in this directory.

#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef HAVE_X11
#include <X11/Xlib.h>
// Xlib's macros collide with common identifiers; we only need the display dims.
#undef None
#undef Status
#undef Success
#undef Bool
#undef True
#undef False
#endif

namespace {

// D455 native profiles that both color and depth support at 30 fps. Larger
// than 848x480 so a near-fullscreen window shows real detail, not upscale mush.
constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kFps = 30;

// Fraction of the screen the window should occupy by default.
constexpr double kDefaultScreenFrac = 0.92;

// rs2 colorizer presets: 0 Jet, 1 Classic, 2 WhiteToBlack, 3 BlackToWhite,
// 4 Bio, 5 Cold, 6 Warm, 7 Quantized, 8 Pattern, 9 Hue
const std::vector<std::pair<int, const char*>> kColormaps = {
    {0, "Jet"}, {2, "WhiteToBlack"}, {3, "BlackToWhite"}, {9, "Hue"}};

// Screen dimensions of the X display we are drawing on, if we can learn them.
bool screen_size(int& w, int& h) {
#ifdef HAVE_X11
  if (Display* d = XOpenDisplay(nullptr)) {
    const int s = DefaultScreen(d);
    w = DisplayWidth(d, s);
    h = DisplayHeight(d, s);
    XCloseDisplay(d);
    return w > 0 && h > 0;
  }
#endif
  return false;
}

cv::Mat frame_to_mat(const rs2::video_frame& f) {
  const int w = f.get_width();
  const int h = f.get_height();
  // Wrap the librealsense buffer, then clone so the Mat outlives the frame.
  switch (f.get_profile().format()) {
    case RS2_FORMAT_BGR8:
      return cv::Mat(h, w, CV_8UC3, const_cast<void*>(f.get_data())).clone();
    case RS2_FORMAT_RGB8: {
      cv::Mat rgb(h, w, CV_8UC3, const_cast<void*>(f.get_data()));
      cv::Mat bgr;
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      return bgr;
    }
    default:
      throw std::runtime_error("unsupported frame format for display");
  }
}

// Text is drawn on the camera-resolution image, which the window then scales
// up, so size the font relative to the image height to keep it legible.
void draw_label(cv::Mat& img, const std::string& text, cv::Point org) {
  const double fs = std::max(0.5, img.rows / 720.0 * 0.8);
  const int th = std::max(1, static_cast<int>(fs * 2));
  cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(0, 0, 0),
              th + 2, cv::LINE_AA);
  cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, fs,
              cv::Scalar(255, 255, 255), th, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) try {
  bool headless = false;
  bool fullscreen = false;
  double frac = kDefaultScreenFrac;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--headless") {
      headless = true;
    } else if (a == "--fullscreen") {
      fullscreen = true;
    } else if (a == "--scale" && i + 1 < argc) {
      frac = std::clamp(std::atof(argv[++i]), 0.1, 1.0);
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--headless] [--fullscreen] [--scale 0.1..1.0]\n";
      return 2;
    }
  }

  rs2::context ctx;
  auto devices = ctx.query_devices();
  if (devices.size() == 0) {
    std::cerr << "No RealSense device found. Is the D455 plugged in?\n";
    return 1;
  }
  auto dev = devices.front();
  std::cout << "Device : " << dev.get_info(RS2_CAMERA_INFO_NAME) << "\n"
            << "Serial : " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << "\n"
            << "FW     : " << dev.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << "\n"
            << "USB    : " << dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)
            << std::endl;

  rs2::pipeline pipe;
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, kWidth, kHeight, RS2_FORMAT_BGR8, kFps);
  cfg.enable_stream(RS2_STREAM_DEPTH, kWidth, kHeight, RS2_FORMAT_Z16, kFps);

  rs2::pipeline_profile profile;
  try {
    profile = pipe.start(cfg);
  } catch (const rs2::error& e) {
    std::cerr << "Requested " << kWidth << "x" << kHeight << "@" << kFps
              << " failed (" << e.what() << "); falling back to defaults.\n";
    rs2::config fallback;
    fallback.enable_stream(RS2_STREAM_COLOR);
    fallback.enable_stream(RS2_STREAM_DEPTH);
    profile = pipe.start(fallback);
  }

  const float depth_scale =
      profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
  std::cout << "Depth scale: " << depth_scale << " m/unit" << std::endl;

  int screen_w = 1920, screen_h = 1080;
  const bool have_screen = screen_size(screen_w, screen_h);
  if (!headless) {
    std::cout << "Screen : " << screen_w << "x" << screen_h
              << (have_screen ? "" : " (assumed - X query failed)")
              << std::endl;
  }

  rs2::align align_to_color(RS2_STREAM_COLOR);
  rs2::colorizer colorizer;
  size_t colormap_idx = 0;
  colorizer.set_option(RS2_OPTION_COLOR_SCHEME,
                       static_cast<float>(kColormaps[colormap_idx].first));

  bool aligned = true;
  bool side_by_side = true;
  bool window_was_visible = false;
  cv::Size sized_for(0, 0);  // canvas size the window was last fitted to

  const std::string win = "D455  |  RGB + Depth";
  if (!headless) {
    // WINDOW_NORMAL lets us resize freely; the backend scales the image to the
    // window, so the camera keeps running at its native resolution.
    cv::namedWindow(win, cv::WINDOW_NORMAL);
    if (fullscreen) {
      cv::setWindowProperty(win, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    }
  }

  auto t_prev = std::chrono::steady_clock::now();
  double fps = 0.0;
  int snapshot_n = 0;
  int frames = 0;

  while (true) {
    rs2::frameset fs = pipe.wait_for_frames();
    if (aligned) {
      fs = align_to_color.process(fs);
    }

    rs2::video_frame color = fs.get_color_frame();
    rs2::depth_frame depth = fs.get_depth_frame();
    if (!color || !depth) {
      continue;
    }

    cv::Mat color_img = frame_to_mat(color);
    cv::Mat depth_img = frame_to_mat(colorizer.process(depth).as<rs2::video_frame>());

    // Depth reading at the image center, in meters (0 == no return).
    const float center_m =
        depth.get_distance(depth.get_width() / 2, depth.get_height() / 2);

    const auto t_now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(t_now - t_prev).count();
    t_prev = t_now;
    if (dt > 0.0) {
      fps = (fps == 0.0) ? 1.0 / dt : 0.9 * fps + 0.1 / dt;
    }
    ++frames;

    char buf[160];
    std::snprintf(buf, sizeof(buf), "RGB  %dx%d  %.1f fps", color_img.cols,
                  color_img.rows, fps);
    draw_label(color_img, buf, {14, 38});
    std::snprintf(buf, sizeof(buf), "Depth  %s  [%s]  center %.3f m",
                  aligned ? "aligned->color" : "raw",
                  kColormaps[colormap_idx].second, center_m);
    draw_label(depth_img, buf, {14, 38});
    draw_label(depth_img, "q quit   a align   c colormap   l layout   f fullscreen   s snapshot",
               {14, depth_img.rows - 18});

    // Center crosshair on both panes so the readout has a visible target.
    for (cv::Mat* m : {&color_img, &depth_img}) {
      const cv::Point c(m->cols / 2, m->rows / 2);
      cv::drawMarker(*m, c, cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 28, 3);
    }

    cv::Mat canvas;
    if (side_by_side) {
      if (depth_img.rows != color_img.rows) {
        cv::resize(depth_img, depth_img,
                   cv::Size(depth_img.cols * color_img.rows / depth_img.rows,
                            color_img.rows));
      }
      cv::hconcat(color_img, depth_img, canvas);
    } else {
      if (depth_img.cols != color_img.cols) {
        cv::resize(depth_img, depth_img,
                   cv::Size(color_img.cols,
                            depth_img.rows * color_img.cols / depth_img.cols));
      }
      cv::vconcat(color_img, depth_img, canvas);
    }

    if (headless) {
      // Sanity mode for SSH: prove frames are flowing, then write one PNG.
      if (frames == 30) {
        cv::imwrite("/tmp/d455_headless.png", canvas);
        std::cout << "Captured 30 frames at " << fps
                  << " fps; wrote /tmp/d455_headless.png" << std::endl;
        break;
      }
      continue;
    }

    // Fit the window to the screen the first time, and again whenever the
    // canvas aspect changes (layout toggle).
    if (!fullscreen && canvas.size() != sized_for) {
      sized_for = canvas.size();
      const double s = std::min(frac * screen_w / canvas.cols,
                                frac * screen_h / canvas.rows);
      const int win_w = static_cast<int>(canvas.cols * s);
      const int win_h = static_cast<int>(canvas.rows * s);
      cv::resizeWindow(win, win_w, win_h);
      cv::moveWindow(win, (screen_w - win_w) / 2, (screen_h - win_h) / 2);
      std::cout << "Window : " << win_w << "x" << win_h << " (canvas "
                << canvas.cols << "x" << canvas.rows << ", "
                << static_cast<int>(s * 100) << "% scale)" << std::endl;
    }

    cv::imshow(win, canvas);
    const int key = cv::waitKey(1) & 0xFF;
    if (key == 'q' || key == 27) {
      break;
    } else if (key == 'a') {
      aligned = !aligned;
    } else if (key == 'l') {
      side_by_side = !side_by_side;
    } else if (key == 'f') {
      fullscreen = !fullscreen;
      cv::setWindowProperty(
          win, cv::WND_PROP_FULLSCREEN,
          fullscreen ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
      sized_for = cv::Size(0, 0);  // refit on the way back out of fullscreen
    } else if (key == 'c') {
      colormap_idx = (colormap_idx + 1) % kColormaps.size();
      colorizer.set_option(RS2_OPTION_COLOR_SCHEME,
                           static_cast<float>(kColormaps[colormap_idx].first));
    } else if (key == 's') {
      const std::string path =
          "d455_snapshot_" + std::to_string(snapshot_n++) + ".png";
      cv::imwrite(path, canvas);
      std::cout << "Saved " << path << std::endl;
    }

    // Window-closed detection: the GTK backend reports the window as not yet
    // visible for the first frame or two, so only trust a 0 after we have seen
    // a 1 at least once.
    const bool visible = cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) >= 1;
    if (visible) {
      window_was_visible = true;
    } else if (window_was_visible) {
      break;  // closed via the title bar
    }
  }

  pipe.stop();
  return 0;
} catch (const rs2::error& e) {
  std::cerr << "RealSense error in " << e.get_failed_function() << "("
            << e.get_failed_args() << "): " << e.what() << "\n";
  return 1;
} catch (const std::exception& e) {
  std::cerr << "Error: " << e.what() << "\n";
  return 1;
}

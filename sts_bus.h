// sts_bus.h - minimal read-only client for the Feetech STS/SCS servo bus used
// by the SO-101 arm (STS3215 servos behind a CH343 USB-serial bridge).
//
// This header deliberately implements only PING and READ. Nothing here writes
// to a servo register, so it cannot change torque, limits, IDs, or commanded
// position - attaching it to a live arm is safe.

#pragma once

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace sts {

// STS3215 control-table addresses (the contiguous "present state" block).
constexpr uint8_t kPresentPosition = 56;  // 2 bytes, 0..4095
constexpr uint8_t kPresentBlockLen = 8;   // pos, speed, load, voltage, temp

constexpr uint8_t kInstPing = 0x01;
constexpr uint8_t kInstRead = 0x02;

constexpr int kTicksPerRev = 4096;

// One servo's decoded present state.
struct Present {
  uint16_t position = 0;   // raw ticks, 0..4095
  int16_t speed = 0;       // signed, raw units
  int16_t load = 0;        // signed, 0..1000 magnitude (~0.1% of max torque)
  uint8_t voltage = 0;     // decivolts
  uint8_t temperature = 0; // degrees C
};

class Bus {
 public:
  Bus() = default;
  ~Bus() { close_port(); }

  Bus(const Bus&) = delete;
  Bus& operator=(const Bus&) = delete;

  bool open_port(const std::string& path, int baud, std::string* err) {
    close_port();
    fd_ = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      if (err) *err = path + ": " + std::strerror(errno);
      return false;
    }
    termios tio{};
    if (tcgetattr(fd_, &tio) != 0) {
      if (err) *err = "tcgetattr: " + std::string(std::strerror(errno));
      close_port();
      return false;
    }
    cfmakeraw(&tio);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~PARENB;
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 0;
    speed_t sp;
    switch (baud) {
      case 1000000: sp = B1000000; break;
      case 500000:  sp = B500000;  break;
      case 250000:  sp = B230400;  break;  // closest standard rate
      case 115200:  sp = B115200;  break;
      default:
        if (err) *err = "unsupported baud " + std::to_string(baud);
        close_port();
        return false;
    }
    cfsetispeed(&tio, sp);
    cfsetospeed(&tio, sp);
    if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
      if (err) *err = "tcsetattr: " + std::string(std::strerror(errno));
      close_port();
      return false;
    }
    tcflush(fd_, TCIOFLUSH);
    return true;
  }

  void close_port() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  bool is_open() const { return fd_ >= 0; }

  // Returns true if the servo answers a PING.
  bool ping(uint8_t id) {
    std::vector<uint8_t> pkt{0xFF, 0xFF, id, 0x02, kInstPing};
    pkt.push_back(checksum(pkt));
    std::vector<uint8_t> resp;
    return transact(pkt, id, 0, &resp);
  }

  // Reads `len` bytes starting at `addr`. `out` must hold at least `len` bytes.
  bool read_block(uint8_t id, uint8_t addr, uint8_t len, uint8_t* out) {
    std::vector<uint8_t> pkt{0xFF, 0xFF, id, 0x04, kInstRead, addr, len};
    pkt.push_back(checksum(pkt));
    std::vector<uint8_t> resp;
    if (!transact(pkt, id, len, &resp)) {
      return false;
    }
    std::memcpy(out, resp.data(), len);
    return true;
  }

  // Reads the present-state block and decodes it.
  bool read_present(uint8_t id, Present* p) {
    uint8_t b[kPresentBlockLen];
    if (!read_block(id, kPresentPosition, kPresentBlockLen, b)) {
      return false;
    }
    p->position = static_cast<uint16_t>(b[0] | (b[1] << 8));
    p->speed = sign_magnitude(static_cast<uint16_t>(b[2] | (b[3] << 8)), 15);
    p->load = sign_magnitude(static_cast<uint16_t>(b[4] | (b[5] << 8)), 10);
    p->voltage = b[6];
    p->temperature = b[7];
    return true;
  }

 private:
  static uint8_t checksum(const std::vector<uint8_t>& pkt) {
    unsigned sum = 0;
    for (size_t i = 2; i < pkt.size(); ++i) {
      sum += pkt[i];
    }
    return static_cast<uint8_t>(~sum & 0xFF);
  }

  // Feetech encodes speed and load as sign-magnitude, not two's complement.
  static int16_t sign_magnitude(uint16_t v, int sign_bit) {
    const uint16_t mask = static_cast<uint16_t>((1u << sign_bit) - 1u);
    const int16_t mag = static_cast<int16_t>(v & mask);
    return (v & (1u << sign_bit)) ? static_cast<int16_t>(-mag) : mag;
  }

  // Writes a packet and waits for this servo's status reply.
  // `payload_len` is the number of data bytes expected back (0 for PING).
  bool transact(const std::vector<uint8_t>& pkt, uint8_t id, uint8_t payload_len,
                std::vector<uint8_t>* payload) {
    if (fd_ < 0) {
      return false;
    }
    tcflush(fd_, TCIFLUSH);
    if (::write(fd_, pkt.data(), pkt.size()) != static_cast<ssize_t>(pkt.size())) {
      return false;
    }

    // Status packet: FF FF ID LEN ERR [payload...] CHK
    const size_t want = 6 + payload_len;
    std::vector<uint8_t> buf;
    buf.reserve(want + 16);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);

    while (std::chrono::steady_clock::now() < deadline) {
      pollfd pfd{fd_, POLLIN, 0};
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now())
                            .count();
      if (::poll(&pfd, 1, static_cast<int>(left > 0 ? left : 0)) <= 0) {
        break;
      }
      uint8_t chunk[64];
      const ssize_t n = ::read(fd_, chunk, sizeof(chunk));
      if (n > 0) {
        buf.insert(buf.end(), chunk, chunk + n);
      }

      // The bus is half duplex, so tolerate a leading echo of our own packet by
      // scanning for the header rather than assuming it starts at byte 0.
      for (size_t i = 0; i + want <= buf.size(); ++i) {
        if (buf[i] == 0xFF && buf[i + 1] == 0xFF && buf[i + 2] == id &&
            buf[i + 3] == payload_len + 2) {
          const uint8_t* frame = buf.data() + i;
          unsigned sum = 0;
          for (size_t k = 2; k < want - 1; ++k) {
            sum += frame[k];
          }
          if (static_cast<uint8_t>(~sum & 0xFF) != frame[want - 1]) {
            continue;  // bad checksum, keep scanning
          }
          payload->assign(frame + 5, frame + 5 + payload_len);
          last_error_flags_ = frame[4];
          return true;
        }
      }
    }
    return false;
  }

  static constexpr int kTimeoutMs = 20;

  int fd_ = -1;
  uint8_t last_error_flags_ = 0;
};

}  // namespace sts

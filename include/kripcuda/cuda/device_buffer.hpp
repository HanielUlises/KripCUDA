#pragma once

#include "kripcuda/cuda/error.hpp"

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace kripcuda {

/// Owning, move-only handle to a linear device allocation.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    explicit DeviceBuffer(std::span<const T> host) {
        allocate(host.size());
        copyFromHost(host);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
    [[nodiscard]] std::size_t sizeBytes() const noexcept { return count_ * sizeof(T); }

    void copyFromHost(std::span<const T> host, cudaStream_t stream = nullptr) {
        if (host.size() > count_) {
            throw std::out_of_range("DeviceBuffer::copyFromHost: source exceeds capacity");
        }
        KRIPCUDA_CHECK(cudaMemcpyAsync(data_, host.data(), host.size_bytes(),
                                       cudaMemcpyHostToDevice, stream));
    }

    void copyToHost(std::span<T> host, cudaStream_t stream = nullptr) const {
        if (host.size() > count_) {
            throw std::out_of_range("DeviceBuffer::copyToHost: destination exceeds source");
        }
        KRIPCUDA_CHECK(cudaMemcpyAsync(host.data(), data_, host.size_bytes(),
                                       cudaMemcpyDeviceToHost, stream));
    }

    [[nodiscard]] std::vector<T> toHost(cudaStream_t stream = nullptr) const {
        std::vector<T> host(count_);
        copyToHost(host, stream);
        KRIPCUDA_CHECK(cudaStreamSynchronize(stream));
        return host;
    }

    /// Device-to-device copy into a fresh allocation.
    [[nodiscard]] DeviceBuffer clone(cudaStream_t stream = nullptr) const {
        DeviceBuffer copy(count_);
        if (count_ != 0) {
            KRIPCUDA_CHECK(cudaMemcpyAsync(copy.data_, data_, sizeBytes(),
                                           cudaMemcpyDeviceToDevice, stream));
        }
        return copy;
    }

    void fillBytes(int value, cudaStream_t stream = nullptr) {
        KRIPCUDA_CHECK(cudaMemsetAsync(data_, value, sizeBytes(), stream));
    }

private:
    void allocate(std::size_t count) {
        if (count == 0) {
            return;
        }
        KRIPCUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
        count_ = count;
    }

    void release() noexcept {
        if (data_ != nullptr) {
            // Destructors must not throw; a failing free during shutdown is
            // reported by the next checked CUDA call on this thread.
            static_cast<void>(cudaFree(data_));
            data_ = nullptr;
        }
        count_ = 0;
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};

/// Creates a device buffer holding a copy of a host range.
template <typename T>
[[nodiscard]] DeviceBuffer<T> makeDeviceBuffer(std::span<const T> host) {
    return DeviceBuffer<T>(host);
}

} // namespace kripcuda

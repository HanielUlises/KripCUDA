#pragma once

#include "kripcuda/cuda/error.cuh"

#include <utility>

namespace kripcuda {

/// Owning, move-only CUDA stream.
class Stream {
public:
    Stream() { KRIPCUDA_CHECK(cudaStreamCreate(&stream_)); }

    explicit Stream(unsigned flags) { KRIPCUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags)); }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            destroy();
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    ~Stream() { destroy(); }

    [[nodiscard]] cudaStream_t handle() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; } // NOLINT(google-explicit-constructor)

    void synchronize() const { KRIPCUDA_CHECK(cudaStreamSynchronize(stream_)); }

    [[nodiscard]] bool queryIdle() const {
        const cudaError_t status = cudaStreamQuery(stream_);
        if (status == cudaSuccess) {
            return true;
        }
        if (status == cudaErrorNotReady) {
            return false;
        }
        throw CudaError(status, "cudaStreamQuery", __FILE__, __LINE__);
    }

private:
    void destroy() noexcept {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_ = nullptr;
};

/// Owning CUDA event, used both for cross-stream ordering and for device-side
/// timing that excludes host overhead.
class Event {
public:
    Event() { KRIPCUDA_CHECK(cudaEventCreate(&event_)); }

    explicit Event(unsigned flags) { KRIPCUDA_CHECK(cudaEventCreateWithFlags(&event_, flags)); }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    Event(Event&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            destroy();
            event_ = std::exchange(other.event_, nullptr);
        }
        return *this;
    }

    ~Event() { destroy(); }

    [[nodiscard]] cudaEvent_t handle() const noexcept { return event_; }

    void record(cudaStream_t stream = nullptr) const {
        KRIPCUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() const { KRIPCUDA_CHECK(cudaEventSynchronize(event_)); }

    /// Milliseconds between two recorded events; both must have completed.
    [[nodiscard]] static float elapsedMilliseconds(const Event& start, const Event& end) {
        float milliseconds = 0.0F;
        KRIPCUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.event_, end.event_));
        return milliseconds;
    }

private:
    void destroy() noexcept {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
            event_ = nullptr;
        }
    }

    cudaEvent_t event_ = nullptr;
};

/// Scoped device-side timer. Records on construction and on destruction and
/// accumulates the interval into the supplied counter.
class ScopedTimer {
public:
    ScopedTimer(float& accumulatorMilliseconds, cudaStream_t stream)
        : accumulator_(accumulatorMilliseconds), stream_(stream) {
        start_.record(stream_);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    ~ScopedTimer() {
        // Timing must never propagate an exception out of a destructor; a
        // failure here leaves the accumulator untouched.
        if (cudaEventRecord(end_.handle(), stream_) != cudaSuccess) {
            return;
        }
        if (cudaEventSynchronize(end_.handle()) != cudaSuccess) {
            return;
        }
        float milliseconds = 0.0F;
        if (cudaEventElapsedTime(&milliseconds, start_.handle(), end_.handle()) == cudaSuccess) {
            accumulator_ += milliseconds;
        }
    }

private:
    float& accumulator_;
    cudaStream_t stream_;
    Event start_;
    Event end_;
};

} // namespace kripcuda

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace atlc {
    class dynamic_bitset {
    public:
        // --- constructors ------------------------------------------------------
        explicit dynamic_bitset(std::size_t bit_count = 0, bool value = false)
            : bits_(bit_count),
            data_((bit_count + bits_per_byte - 1) / bits_per_byte,
                    value ? 0xFF : 0x00) {
            if (value) trim_unused_bits();
        }

        // --- element access ----------------------------------------------------
        bool operator[](std::size_t pos) const { return test(pos); }

        bool test(std::size_t pos) const {
            check_range(pos);
            return (data_[pos / bits_per_byte] >> (pos % bits_per_byte)) & 1U;
        }

        // --- modifiers ---------------------------------------------------------
        void set(std::size_t pos, bool value = true) {
            check_range(pos);
            if (value)
                data_[pos / bits_per_byte] |=  (uint8_t(1) << (pos % bits_per_byte));
            else
                data_[pos / bits_per_byte] &= ~(uint8_t(1) << (pos % bits_per_byte));
        }

        void reset(std::size_t pos)           { set(pos, false); }
        void flip(std::size_t pos) {
            check_range(pos);
            data_[pos / bits_per_byte] ^= (uint8_t(1) << (pos % bits_per_byte));
        }

        void reset_all() { std::fill(data_.begin(), data_.end(), 0); }
        void set_all()   { std::fill(data_.begin(), data_.end(), 0xFF); trim_unused_bits(); }
        void flip_all()  { for (auto &b : data_) b = ~b; trim_unused_bits(); }

        // --- capacity ----------------------------------------------------------
        std::size_t size()  const noexcept { return bits_; }
        bool         empty() const noexcept { return bits_ == 0; }

        // --- operations --------------------------------------------------------
        std::size_t count() const noexcept {
            std::size_t c = 0;
            for (uint8_t byte : data_) c += byte_bit_count(byte);
            return c;
        }
        bool any()  const noexcept { return count() != 0; }
        bool none() const noexcept { return !any(); }

        void resize(std::size_t new_bits, bool value = false) {
            bits_ = new_bits;
            const std::size_t new_bytes = (new_bits + bits_per_byte - 1) / bits_per_byte;
            data_.resize(new_bytes, value ? 0xFF : 0x00);
            if (value) trim_unused_bits();
        }

        void push_back(bool value) {
            resize(bits_ + 1, false);
            set(bits_ - 1, value);
        }

        // --- string conversion -------------------------------------------------
        std::string to_string(char zero = '0', char one = '1') const {
            std::string s;
            s.reserve(bits_);
            for (std::size_t i = 0; i < bits_; ++i)
                s.push_back(test(bits_ - 1 - i) ? one : zero);
            return s;
        }

    private:
        // --- helpers -----------------------------------------------------------
        static std::size_t byte_bit_count(uint8_t byte) noexcept {
            // 8‑bit popcount (Hacker's Delight)
            byte = byte - ((byte >> 1) & 0x55);
            byte = (byte & 0x33) + ((byte >> 2) & 0x33);
            return (((byte + (byte >> 4)) & 0x0F) * 0x01);
        }

        void check_range(std::size_t pos) const {
            if (pos >= bits_) throw std::out_of_range("dynamic_bitset: index out of range");
        }

        void trim_unused_bits() {
            const std::size_t unused = data_.size() * bits_per_byte - bits_;
            if (unused) {
                const uint8_t mask = uint8_t(0xFFu >> unused);
                data_.back() &= mask; // clear the padding bits
            }
        }

        // --- data members ------------------------------------------------------
        static constexpr std::size_t bits_per_byte = 8;
        std::size_t bits_;          // logical bit count
        std::vector<uint8_t> data_; // packed storage
    };
}
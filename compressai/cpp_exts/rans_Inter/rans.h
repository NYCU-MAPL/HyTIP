/* Copyright 2020 InterDigital Communications, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <condition_variable>
#include <list>
#include <thread>
#include <vector>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif

#include "rans_byte.h"

#ifdef _MSC_VER
#pragma warning(default : 4244)
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

struct RansSymbol {
  uint16_t start;
  uint16_t range; // range for normal coding and 0 for bypass coding
};

enum class WorkType {
  Encode,
  Flush,
};

struct PendingTask {
  WorkType workType;
  std::shared_ptr<std::vector<int16_t>> symbols;
  std::shared_ptr<std::vector<int16_t>> indexes;
  int cdf_group_index{0};
};

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class MLCodec_RansEncoder_InterLib {
public:
  MLCodec_RansEncoder_InterLib() {}
  virtual ~MLCodec_RansEncoder_InterLib() = default;

  MLCodec_RansEncoder_InterLib(const MLCodec_RansEncoder_InterLib &) = delete;
  MLCodec_RansEncoder_InterLib(MLCodec_RansEncoder_InterLib &&) = delete;
  MLCodec_RansEncoder_InterLib &operator=(const MLCodec_RansEncoder_InterLib &) = delete;
  MLCodec_RansEncoder_InterLib &operator=(MLCodec_RansEncoder_InterLib &&) = delete;

  virtual void
  encode_with_indexes(const std::shared_ptr<std::vector<int16_t>> symbols,
                      const std::shared_ptr<std::vector<int16_t>> indexes,
                      const int cdf_group_index);
  virtual void flush();
  virtual std::vector<uint8_t> get_encoded_stream();
  virtual void reset();
  virtual int
  add_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
          const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
          const std::shared_ptr<std::vector<int32_t>> offsets);
  virtual void empty_cdf_buffer();

private:
  std::vector<RansSymbol> _syms;
  std::vector<uint8_t> _stream;

  std::vector<std::shared_ptr<std::vector<std::vector<RansSymbol>>>>
      _ransSymbols;
  std::vector<std::shared_ptr<std::vector<int32_t>>> _cdfs_sizes;
  std::vector<std::shared_ptr<std::vector<int32_t>>> _offsets;
};

class MLCodec_RansEncoder_InterLibMultiThread : public MLCodec_RansEncoder_InterLib {
public:
  MLCodec_RansEncoder_InterLibMultiThread();
  virtual ~MLCodec_RansEncoder_InterLibMultiThread();

  virtual void
  encode_with_indexes(const std::shared_ptr<std::vector<int16_t>> symbols,
                      const std::shared_ptr<std::vector<int16_t>> indexes,
                      const int cdf_group_index) override;
  virtual void flush() override;
  virtual std::vector<uint8_t> get_encoded_stream() override;
  virtual void reset() override;

  void worker();

private:
  bool m_finish;
  bool m_result_ready;
  std::thread m_thread;
  std::mutex m_mutex_result;
  std::mutex m_mutex_pending;
  std::condition_variable m_cv_pending;
  std::condition_variable m_cv_result;
  std::list<PendingTask> m_pending;
};

class MLCodec_RansDecoder_InterLib {
public:
  MLCodec_RansDecoder_InterLib() {}
  virtual ~MLCodec_RansDecoder_InterLib() = default;

  MLCodec_RansDecoder_InterLib(const MLCodec_RansDecoder_InterLib &) = delete;
  MLCodec_RansDecoder_InterLib(MLCodec_RansDecoder_InterLib &&) = delete;
  MLCodec_RansDecoder_InterLib &operator=(const MLCodec_RansDecoder_InterLib &) = delete;
  MLCodec_RansDecoder_InterLib &operator=(MLCodec_RansDecoder_InterLib &&) = delete;

  void set_stream(const std::shared_ptr<std::vector<uint8_t>> encoded);

  std::vector<int16_t>
  decode_stream(const std::shared_ptr<std::vector<int16_t>> indexes,
                const int cdf_group_index);

  virtual int
  add_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
          const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
          const std::shared_ptr<std::vector<int32_t>> offsets);
  virtual void empty_cdf_buffer();

private:
  RansState _rans;
  uint8_t *_ptr8;
  std::shared_ptr<std::vector<uint8_t>> _stream;

  std::vector<std::shared_ptr<std::vector<std::vector<int32_t>>>> _cdfs;
  std::vector<std::shared_ptr<std::vector<int32_t>>> _cdfs_sizes;
  std::vector<std::shared_ptr<std::vector<int32_t>>> _offsets;
};

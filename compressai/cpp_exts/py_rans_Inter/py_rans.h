// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "rans.h"
#include <memory>
#include <pybind11/numpy.h>

namespace py = pybind11;

// the classes in this file only perform the type conversion
// from python type (numpy) to C++ type (vector)
class MLCodec_RansEncoder_Inter {
public:
  MLCodec_RansEncoder_Inter(bool multiThread, int streamPart);

  MLCodec_RansEncoder_Inter(const MLCodec_RansEncoder_Inter &) = delete;
  MLCodec_RansEncoder_Inter(MLCodec_RansEncoder_Inter &&) = delete;
  MLCodec_RansEncoder_Inter &operator=(const MLCodec_RansEncoder_Inter &) = delete;
  MLCodec_RansEncoder_Inter &operator=(MLCodec_RansEncoder_Inter &&) = delete;

  void encode_with_indexes(const py::array_t<int16_t> &symbols,
                           const py::array_t<int16_t> &indexes,
                           const int cdf_group_index);
  void flush();
  py::array_t<uint8_t> get_encoded_stream();
  void reset();
  int add_cdf(const py::array_t<int32_t> &cdfs,
              const py::array_t<int32_t> &cdfs_sizes,
              const py::array_t<int32_t> &offsets);
  void empty_cdf_buffer();

private:
  std::vector<std::shared_ptr<MLCodec_RansEncoder_InterLib>> m_encoders;
};

class MLCodec_RansDecoder_Inter {
public:
  MLCodec_RansDecoder_Inter(int streamPart);

  MLCodec_RansDecoder_Inter(const MLCodec_RansDecoder_Inter &) = delete;
  MLCodec_RansDecoder_Inter(MLCodec_RansDecoder_Inter &&) = delete;
  MLCodec_RansDecoder_Inter &operator=(const MLCodec_RansDecoder_Inter &) = delete;
  MLCodec_RansDecoder_Inter &operator=(MLCodec_RansDecoder_Inter &&) = delete;

  void set_stream(const py::array_t<uint8_t> &);

  py::array_t<int16_t> decode_stream(const py::array_t<int16_t> &indexes,
                                     const int cdf_group_index);
  int add_cdf(const py::array_t<int32_t> &cdfs,
              const py::array_t<int32_t> &cdfs_sizes,
              const py::array_t<int32_t> &offsets);
  void empty_cdf_buffer();

private:
  std::vector<std::shared_ptr<MLCodec_RansDecoder_InterLib>> m_decoders;
};

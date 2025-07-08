// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "rans.h"
#include <memory>
#include <pybind11/numpy.h>

namespace py = pybind11;

// the classes in this file only perform the type conversion
// from python type (numpy) to C++ type (vector)
class MLCodec_RansEncoder_Intra {
public:
  MLCodec_RansEncoder_Intra(bool multiThread, int streamPart);

  MLCodec_RansEncoder_Intra(const MLCodec_RansEncoder_Intra &) = delete;
  MLCodec_RansEncoder_Intra(MLCodec_RansEncoder_Intra &&) = delete;
  MLCodec_RansEncoder_Intra &operator=(const MLCodec_RansEncoder_Intra &) = delete;
  MLCodec_RansEncoder_Intra &operator=(MLCodec_RansEncoder_Intra &&) = delete;

  void encode_with_indexes(const py::array_t<int16_t> &symbols,
                           const py::array_t<int16_t> &indexes,
                           const py::array_t<int32_t> &cdfs,
                           const py::array_t<int32_t> &cdfs_sizes,
                           const py::array_t<int32_t> &offsets);
  void flush();
  py::array_t<uint8_t> get_encoded_stream();
  void reset();

private:
  std::vector<std::shared_ptr<MLCodec_RansEncoder_IntraLib>> m_encoders;
};

class MLCodec_RansDecoder_Intra {
public:
  MLCodec_RansDecoder_Intra(int streamPart);

  MLCodec_RansDecoder_Intra(const MLCodec_RansDecoder_Intra &) = delete;
  MLCodec_RansDecoder_Intra(MLCodec_RansDecoder_Intra &&) = delete;
  MLCodec_RansDecoder_Intra &operator=(const MLCodec_RansDecoder_Intra &) = delete;
  MLCodec_RansDecoder_Intra &operator=(MLCodec_RansDecoder_Intra &&) = delete;

  void set_stream(const py::array_t<uint8_t> &);

  py::array_t<int16_t> decode_stream(const py::array_t<int16_t> &indexes,
                                     const py::array_t<int32_t> &cdfs,
                                     const py::array_t<int32_t> &cdfs_sizes,
                                     const py::array_t<int32_t> &offsets);

private:
  std::vector<std::shared_ptr<MLCodec_RansDecoder_IntraLib>> m_decoders;
};

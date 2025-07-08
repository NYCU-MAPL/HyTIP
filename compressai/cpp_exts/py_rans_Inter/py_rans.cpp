// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_rans.h"

#include <future>
#include <vector>

namespace py = pybind11;

MLCodec_RansEncoder_Inter::MLCodec_RansEncoder_Inter(bool multiThread, int streamPart) {
  bool useMultiThread = multiThread || streamPart > 1;
  for (int i = 0; i < streamPart; i++) {
    if (useMultiThread) {
      m_encoders.push_back(std::make_shared<MLCodec_RansEncoder_InterLibMultiThread>());
    } else {
      m_encoders.push_back(std::make_shared<MLCodec_RansEncoder_InterLib>());
    }
  }
}

void MLCodec_RansEncoder_Inter::encode_with_indexes(const py::array_t<int16_t> &symbols,
                                      const py::array_t<int16_t> &indexes,
                                      const int cdf_group_index) {
  py::buffer_info symbols_buf = symbols.request();
  py::buffer_info indexes_buf = indexes.request();
  int16_t *symbols_ptr = static_cast<int16_t *>(symbols_buf.ptr);
  int16_t *indexes_ptr = static_cast<int16_t *>(indexes_buf.ptr);

  int encoderNum = static_cast<int>(m_encoders.size());
  int symbolSize = static_cast<int>(symbols.size());
  int eachSymbolSize = symbolSize / encoderNum;
  int lastSymbolSize = symbolSize - eachSymbolSize * (encoderNum - 1);
  for (int i = 0; i < encoderNum; i++) {
    int currSymbolSize = i < encoderNum - 1 ? eachSymbolSize : lastSymbolSize;
    int currOffset = i * eachSymbolSize;
    auto copySize = sizeof(int16_t) * currSymbolSize;
    auto vec_symbols = std::make_shared<std::vector<int16_t>>(currSymbolSize);
    memcpy(vec_symbols->data(), symbols_ptr + currOffset, copySize);
    auto vec_indexes = std::make_shared<std::vector<int16_t>>(eachSymbolSize);
    memcpy(vec_indexes->data(), indexes_ptr + currOffset, copySize);
    m_encoders[i]->encode_with_indexes(vec_symbols, vec_indexes,
                                       cdf_group_index);
  }
}

int MLCodec_RansEncoder_Inter::add_cdf(const py::array_t<int32_t> &cdfs,
                         const py::array_t<int32_t> &cdfs_sizes,
                         const py::array_t<int32_t> &offsets) {
  py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
  py::buffer_info offsets_buf = offsets.request();
  int32_t *cdfs_sizes_ptr = static_cast<int32_t *>(cdfs_sizes_buf.ptr);
  int32_t *offsets_ptr = static_cast<int32_t *>(offsets_buf.ptr);

  int cdf_num = static_cast<int>(cdfs_sizes.size());
  auto vec_cdfs_sizes = std::make_shared<std::vector<int32_t>>(cdf_num);
  memcpy(vec_cdfs_sizes->data(), cdfs_sizes_ptr, sizeof(int32_t) * cdf_num);
  auto vec_offsets = std::make_shared<std::vector<int32_t>>(offsets.size());
  memcpy(vec_offsets->data(), offsets_ptr, sizeof(int32_t) * offsets.size());

  int per_vector_size = static_cast<int>(cdfs.size() / cdf_num);
  auto vec_cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(cdf_num);
  auto cdfs_raw = cdfs.unchecked<2>();
  for (int i = 0; i < cdf_num; i++) {
    std::vector<int32_t> t(per_vector_size);
    memcpy(t.data(), cdfs_raw.data(i, 0), sizeof(int32_t) * per_vector_size);
    vec_cdfs->at(i) = t;
  }

  int encoderNum = static_cast<int>(m_encoders.size());
  int cdfIdx = 0;
  for (int i = 0; i < encoderNum; i++) {
    cdfIdx = m_encoders[i]->add_cdf(vec_cdfs, vec_cdfs_sizes, vec_offsets);
  }
  return cdfIdx;
}

void MLCodec_RansEncoder_Inter::empty_cdf_buffer() {
  int encoderNum = static_cast<int>(m_encoders.size());
  for (int i = 0; i < encoderNum; i++) {
    m_encoders[i]->empty_cdf_buffer();
  }
}

void MLCodec_RansEncoder_Inter::flush() {
  for (auto encoder : m_encoders) {
    encoder->flush();
  }
}

py::array_t<uint8_t> MLCodec_RansEncoder_Inter::get_encoded_stream() {
  std::vector<std::vector<uint8_t>> results;
  int maximumSize = 0;
  int totalSize = 0;
  int encoderNumber = static_cast<int>(m_encoders.size());
  for (int i = 0; i < encoderNumber; i++) {
    std::vector<uint8_t> result = m_encoders[i]->get_encoded_stream();
    results.push_back(result);
    int nbytes = static_cast<int>(result.size());
    if (i < encoderNumber - 1 && nbytes > maximumSize) {
      maximumSize = nbytes;
    }
    totalSize += nbytes;
  }

  int overhead = 1;
  int perStreamHeader = maximumSize > 65535 ? 4 : 2;
  if (encoderNumber > 1) {
    overhead += ((encoderNumber - 1) * perStreamHeader);
  }

  py::array_t<uint8_t> stream(totalSize + overhead);
  py::buffer_info stream_buf = stream.request();
  uint8_t *stream_ptr = static_cast<uint8_t *>(stream_buf.ptr);

  uint8_t flag = static_cast<uint8_t>(((encoderNumber - 1) << 4) +
                                      (perStreamHeader == 2 ? 1 : 0));
  memcpy(stream_ptr, &flag, 1);
  for (int i = 0; i < encoderNumber - 1; i++) {
    if (perStreamHeader == 2) {
      uint16_t streamSizes = static_cast<uint16_t>(results[i].size());
      memcpy(stream_ptr + 1 + 2 * i, &streamSizes, 2);
    } else {
      uint32_t streamSizes = static_cast<uint32_t>(results[i].size());
      memcpy(stream_ptr + 1 + 4 * i, &streamSizes, 4);
    }
  }

  int offset = overhead;
  for (int i = 0; i < encoderNumber; i++) {
    int nbytes = static_cast<int>(results[i].size());
    memcpy(stream_ptr + offset, results[i].data(), nbytes);
    offset += nbytes;
  }
  return stream;
}

void MLCodec_RansEncoder_Inter::reset() {
  for (auto encoder : m_encoders) {
    encoder->reset();
  }
}

MLCodec_RansDecoder_Inter::MLCodec_RansDecoder_Inter(int streamPart) {
  for (int i = 0; i < streamPart; i++) {
    m_decoders.push_back(std::make_shared<MLCodec_RansDecoder_InterLib>());
  }
}

void MLCodec_RansDecoder_Inter::set_stream(const py::array_t<uint8_t> &encoded) {
  py::buffer_info encoded_buf = encoded.request();
  uint8_t flag = *(static_cast<uint8_t *>(encoded_buf.ptr));
  int numberOfStreams = (flag >> 4) + 1;

  uint8_t perStreamSizeLength = (flag & 0x0f) == 1 ? 2 : 4;
  std::vector<uint32_t> streamSizes;
  int offset = 1;
  int totalSize = 0;
  for (int i = 0; i < numberOfStreams - 1; i++) {
    uint8_t *currPos = static_cast<uint8_t *>(encoded_buf.ptr) + offset;
    if (perStreamSizeLength == 2) {
      uint16_t streamSize = *(reinterpret_cast<uint16_t *>(currPos));
      offset += 2;
      streamSizes.push_back(streamSize);
      totalSize += streamSize;
    } else {
      uint32_t streamSize = *(reinterpret_cast<uint32_t *>(currPos));
      offset += 4;
      streamSizes.push_back(streamSize);
      totalSize += streamSize;
    }
  }
  streamSizes.push_back(static_cast<int>(encoded.size()) - offset - totalSize);
  for (int i = 0; i < numberOfStreams; i++) {
    auto stream = std::make_shared<std::vector<uint8_t>>(streamSizes[i]);
    memcpy(stream->data(), static_cast<uint8_t *>(encoded_buf.ptr) + offset,
           streamSizes[i]);
    m_decoders[i]->set_stream(stream);
    offset += streamSizes[i];
  }
}

py::array_t<int16_t>
MLCodec_RansDecoder_Inter::decode_stream(const py::array_t<int16_t> &indexes,
                           const int cdf_group_index) {
  py::buffer_info indexes_buf = indexes.request();
  int16_t *indexes_ptr = static_cast<int16_t *>(indexes_buf.ptr);

  int decoderNum = static_cast<int>(m_decoders.size());
  int indexSize = static_cast<int>(indexes.size());
  int eachSymbolSize = indexSize / decoderNum;
  int lastSymbolSize = indexSize - eachSymbolSize * (decoderNum - 1);

  std::vector<std::shared_future<std::vector<int16_t>>> results;

  for (int i = 0; i < decoderNum; i++) {
    int currSymbolSize = i < decoderNum - 1 ? eachSymbolSize : lastSymbolSize;
    int copySize = sizeof(int16_t) * currSymbolSize;
    auto vec_indexes = std::make_shared<std::vector<int16_t>>(currSymbolSize);
    memcpy(vec_indexes->data(), indexes_ptr + i * eachSymbolSize, copySize);

    std::shared_future<std::vector<int16_t>> result =
        std::async(std::launch::async, [=]() {
          return m_decoders[i]->decode_stream(vec_indexes, cdf_group_index);
        });
    results.push_back(result);
  }

  py::array_t<int16_t> output(indexes.size());
  py::buffer_info buf = output.request();
  int offset = 0;
  for (int i = 0; i < decoderNum; i++) {
    std::vector<int16_t> result = results[i].get();
    int resultSize = static_cast<int>(result.size());
    int copySize = sizeof(int16_t) * resultSize;
    memcpy(static_cast<int16_t *>(buf.ptr) + offset, result.data(), copySize);
    offset += resultSize;
  }

  return output;
}

int MLCodec_RansDecoder_Inter::add_cdf(const py::array_t<int32_t> &cdfs,
                         const py::array_t<int32_t> &cdfs_sizes,
                         const py::array_t<int32_t> &offsets) {
  py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
  py::buffer_info offsets_buf = offsets.request();
  int32_t *cdfs_sizes_ptr = static_cast<int32_t *>(cdfs_sizes_buf.ptr);
  int32_t *offsets_ptr = static_cast<int32_t *>(offsets_buf.ptr);

  int cdf_num = static_cast<int>(cdfs_sizes.size());
  auto vec_cdfs_sizes = std::make_shared<std::vector<int32_t>>(cdf_num);
  memcpy(vec_cdfs_sizes->data(), cdfs_sizes_ptr, sizeof(int32_t) * cdf_num);
  auto vec_offsets = std::make_shared<std::vector<int32_t>>(offsets.size());
  memcpy(vec_offsets->data(), offsets_ptr, sizeof(int32_t) * offsets.size());

  int per_vector_size = static_cast<int>(cdfs.size() / cdf_num);
  auto vec_cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(cdf_num);
  auto cdfs_raw = cdfs.unchecked<2>();
  for (int i = 0; i < cdf_num; i++) {
    std::vector<int32_t> t(per_vector_size);
    memcpy(t.data(), cdfs_raw.data(i, 0), sizeof(int32_t) * per_vector_size);
    vec_cdfs->at(i) = t;
  }
  int decoderNum = static_cast<int>(m_decoders.size());
  int cdfIdx = 0;
  for (int i = 0; i < decoderNum; i++) {
    cdfIdx = m_decoders[i]->add_cdf(vec_cdfs, vec_cdfs_sizes, vec_offsets);
  }

  return cdfIdx;
}

void MLCodec_RansDecoder_Inter::empty_cdf_buffer() {
  int decoderNum = static_cast<int>(m_decoders.size());
  for (int i = 0; i < decoderNum; i++) {
    m_decoders[i]->empty_cdf_buffer();
  }
}

PYBIND11_MODULE(MLCodec_rans_Inter, m) {
  m.attr("__name__") = "compressai.MLCodec_rans_Inter";

  m.doc() = "range Asymmetric Numeral System python bindings";

  py::class_<MLCodec_RansEncoder_Inter>(m, "MLCodec_RansEncoder_Inter")
      .def(py::init<bool, int>())
      .def("encode_with_indexes", &MLCodec_RansEncoder_Inter::encode_with_indexes)
      .def("flush", &MLCodec_RansEncoder_Inter::flush)
      .def("get_encoded_stream", &MLCodec_RansEncoder_Inter::get_encoded_stream)
      .def("reset", &MLCodec_RansEncoder_Inter::reset)
      .def("add_cdf", &MLCodec_RansEncoder_Inter::add_cdf)
      .def("empty_cdf_buffer", &MLCodec_RansEncoder_Inter::empty_cdf_buffer);

  py::class_<MLCodec_RansDecoder_Inter>(m, "MLCodec_RansDecoder_Inter")
      .def(py::init<int>())
      .def("set_stream", &MLCodec_RansDecoder_Inter::set_stream)
      .def("decode_stream", &MLCodec_RansDecoder_Inter::decode_stream)
      .def("add_cdf", &MLCodec_RansDecoder_Inter::add_cdf)
      .def("empty_cdf_buffer", &MLCodec_RansDecoder_Inter::empty_cdf_buffer);
}

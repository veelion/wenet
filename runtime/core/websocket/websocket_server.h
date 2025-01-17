// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef WEBSOCKET_WEBSOCKET_SERVER_H_
#define WEBSOCKET_WEBSOCKET_SERVER_H_

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "boost/asio/connect.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/websocket.hpp"

#include "decoder/asr_decoder.h"
#include "frontend/feature_pipeline.h"
#include "utils/log.h"

namespace wenet {

namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http = beast::http;            // from <boost/beast/http.hpp>
namespace websocket = beast::websocket;  // from <boost/beast/websocket.hpp>
namespace asio = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;        // from <boost/asio/ip/tcp.hpp>

class ConnectionHandler {
 public:
  ConnectionHandler(tcp::socket&& socket,
                    std::shared_ptr<FeaturePipelineConfig> feature_config,
                    std::shared_ptr<DecodeOptions> decode_config,
                    std::shared_ptr<DecodeResource> decode_resource_);
  void operator()();

 private:
  void OnSpeechStart();
  void OnSpeechEnd();
  void OnText(const std::string& message);
  void OnFinish();
  void OnSpeechData(const beast::flat_buffer& buffer);
  void OnError(const std::string& message);
  void OnPartialResult(const std::string& result);
  void OnFinalResult(const std::string& result);
  void DecodeThreadFunc();
  std::string SerializeResult(bool finish);

  bool continuous_decoding_ = false;
  int nbest_ = 1;
  websocket::stream<tcp::socket> ws_;
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<DecodeResource> decode_resource_;

  bool got_start_tag_ = false;
  bool got_end_tag_ = false;
  // When endpoint is detected, stop recognition, and stop receiving data.
  bool stop_recognition_ = false;
  std::shared_ptr<FeaturePipeline> feature_pipeline_ = nullptr;
  std::shared_ptr<AsrDecoder> decoder_ = nullptr;
  std::shared_ptr<std::thread> decode_thread_ = nullptr;
};

class WebSocketServer {
 public:
  WebSocketServer(int port,
                  std::shared_ptr<FeaturePipelineConfig> feature_config,
                  std::shared_ptr<DecodeOptions> decode_config,
                  std::shared_ptr<DecodeResource> decode_resource)
      : port_(port),
        feature_config_(std::move(feature_config)),
        decode_config_(std::move(decode_config)),
        decode_resource_(std::move(decode_resource)) {}

  void Start(bool run_batch = false);

 private:
  int port_;
  // The io_context is required for all I/O
  asio::io_context ioc_{1};
  std::shared_ptr<FeaturePipelineConfig> feature_config_;
  std::shared_ptr<DecodeOptions> decode_config_;
  std::shared_ptr<DecodeResource> decode_resource_;
  WENET_DISALLOW_COPY_AND_ASSIGN(WebSocketServer);
};

}  // namespace wenet

#endif  // WEBSOCKET_WEBSOCKET_SERVER_H_

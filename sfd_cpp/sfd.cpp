#include <chrono>

#include "include/sfd.h"

using namespace sfd;

SFD::SFD(const string& model_file,
                       const string& trained_file) {

#ifdef CPU_ONLY
  std::cout << "Using CPU" << std::endl;
  Caffe::set_mode(Caffe::CPU);
#else
  std::cout << "Using GPU" << std::endl;
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];

  //SetMean(mean_file);

  Blob<float>* output_layer = net_->output_blobs()[0];
}


void SFD::set_input_buffer(std::vector<cv::Mat>& input_channels,
    float* input_data, const int height, const int width) {
  for (int i = 0; i < 3; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels.push_back(channel);
    input_data += width * height;
  }
}

std::vector<Detection> SFD::detect(const cv::Mat& image, float threshold) {
  
  cv::Mat img;
  image.convertTo(img, CV_32FC3);
  img = img- cv::Scalar(104.0, 117.0, 123.0);

  int image_width = img.cols;
  int image_height = img.rows;

  Blob<float>* input_blob = net_->input_blobs()[0];
  input_blob->Reshape(1, 3, image_height, image_width);
  net_->Reshape();


  std::vector<cv::Mat> input_channels;
  float * input_data=net_->input_blobs()[0]->mutable_cpu_data();
  set_input_buffer(input_channels, input_data, image_height, image_width);

  cv::split(img, input_channels);

  net_->Forward();

  Blob<float>* reg = net_->output_blobs()[0];
  //Blob<float>* confidence = net_->output_blobs()[1];


  int num_preds = reg->shape(2);
  int num_features =reg->shape(3);
  CHECK_EQ(num_features, 7) << "The model should predict exactly 7 features.";

  const float *pred_data = reg->cpu_data();

  std::vector<Detection> detections;
  for (int i = 0; i < num_preds; i++) {
    float det_conf = pred_data[(num_features * i) + 2];
    float x_min = pred_data[(num_features * i) + 3];
    float y_min = pred_data[(num_features * i) + 4];
    float x_max = pred_data[(num_features * i) + 5];
    float y_max = pred_data[(num_features * i) + 6];
    if (threshold < det_conf) {
      Detection detection(det_conf, x_min, y_min, x_max, y_max);
      detections.push_back(detection);
    }
  }
  return detections;
}

cv::Mat draw_detections(cv::Mat &img, std::vector<Detection> detections) {
  cv::Mat visuallized = img.clone();

  int width = visuallized.cols;
  int height = visuallized.rows;

  for (auto detection : detections) {
    int xmin = int(detection.get_xmin() * (float)width);
    int ymin = int(detection.get_ymin() * (float)height);
    int xmax = int(detection.get_xmax() * (float)width);
    int ymax = int(detection.get_ymax() * (float)height);
    cv::rectangle(visuallized,
            cv::Point(xmin, ymin),
            cv::Point(xmax, ymax),
            cv::Scalar(255, 0, 0));
  }
  return visuallized;
}

string get_filename(string path) {
    std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
    return base_filename;
}

int main(int argc, char** argv) {
  string model_file   = "../../models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt";
  string trained_file = "../../models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel";

  SFD sfd(model_file, trained_file);

  std::vector<string> files;
  cv::glob("../../images/*.jpg", files, false);

  for (auto file : files) {
    std::cout << "---------- Face detection for "
              << file << " ----------" << std::endl;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> detections = sfd.detect(img);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Required time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << std::endl;
    cv::Mat visualized = draw_detections(img, detections);
    cv::imwrite(get_filename(file), visualized);
  }

  return 0;
}
 

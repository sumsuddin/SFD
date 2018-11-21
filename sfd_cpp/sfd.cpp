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


void set_input_buffer(std::vector<cv::Mat>& input_channels,
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
  const float *pred_data = reg->cpu_data();

  for (int i = 0; i < num_preds; i++) {
    float det_conf = pred_data[(num_features * i) + 2];
    std::cout << det_conf << std::endl;
  }

  std::cout << num_preds << " " << num_features << std::endl;

  //std::vector<float> output = Predict(img);
  std::vector<Detection> detections;
  return detections;
}

int main(int argc, char** argv) {
  string model_file   = "../../models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt";
  string trained_file = "../../models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel";

  SFD sfd(model_file, trained_file);

  string file = "../../faces.jpg";

  std::cout << "---------- Face detection for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Detection> detections = sfd.detect(img);
}
 

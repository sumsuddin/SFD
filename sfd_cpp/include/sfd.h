#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


#include "detection.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

namespace sfd {

	class SFD {
	 public:
	  SFD(const string& model_file,
	             const string& trained_file);

	  std::vector<Detection> detect(const cv::Mat& img, const float threshold=0.5);

	 private:

      void set_input_buffer(std::vector<cv::Mat>& input_channels, float* input_data, const int height, const int width);
	  //void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

	 private:
	  shared_ptr<Net<float> > net_;
	};
} 

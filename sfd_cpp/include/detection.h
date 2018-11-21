
namespace sfd {

	class Detection {
	public:

		float get_confidence();
		float get_xmin();
		float get_ymin();
		float get_xmax();
		float get_ymax();

		Detection(const float confidence, const float x_min, const float y_min, const float x_max, const float y_max);

	private:
		float confidence;
		float x_min;
		float y_min;
		float x_max;
		float y_max;
	};
} 

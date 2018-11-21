 

#include "include/detection.h"

namespace sfd {

	Detection::Detection(const float confidence,
		const float x_min,
		const float y_min,
		const float x_max,
		const float y_max) {

		this->confidence = confidence;
		this->x_min = x_min;
		this->y_min = y_min;
		this->x_max = x_max;
		this->y_max = y_max;
	}

		float Detection::get_confidence() { return confidence;}
		float Detection::get_xmin() { return x_min;}
		float Detection::get_ymin() { return y_min;}
		float Detection::get_xmax() { return x_max;}
		float Detection::get_ymax() { return y_max;}
}
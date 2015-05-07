#pragma once
#include <cstdint>

struct BilateralConfig {
	int radius;
	float range_sigma, color_sigma;
	void UseDefault()
	{
		range_sigma = 20.0f;
		color_sigma = 10.0f;
		radius = 20;
	}
};

void bilateral_cxx(const uint8_t *in, uint8_t *out, const BilateralConfig config, const int w, const int h);
void bilateral_ocl(const uint8_t *in, uint8_t *out, const BilateralConfig config, const int w, const int h);

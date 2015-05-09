#include "bilateral.h"
#include "global.h"
#include "cl_helper.h"
#include "cl_helper.h"
#include <glog/logging.h>
#include <cmath>
#include <memory>
using std::exp;
using std::unique_ptr;
using namespace google;

template <class SignedIntType> SignedIntType ClampToUint8(SignedIntType x)
{
	const SignedIntType mask = 0xff;
	return (x&~mask)? ((~x)>>(sizeof(SignedIntType)*8-1) & mask): x;
}

static unique_ptr<float[]> GenerateGaussianTable(const float sigma, const int length)
{
	unique_ptr<float[]> table(new float[length]);
	const float denominator_inverse = -1.0f / (2.0f * sigma * sigma);
	for (int i = 0; i < length; ++i) {
		table.get()[i] = exp(i*i*denominator_inverse);
	}
	return std::move(table);
}

void bilateral_cxx(const uint8_t *in, uint8_t *out, const BilateralConfig config, const int w, const int h)
{
	const int r = config.radius;
	if (w <= 2*r || h <= 2*r) {
		LOG(WARNING) << "No work to do";
		return;
	}
	auto range_gaussian_table = GenerateGaussianTable(config.range_sigma, r+1);
	auto color_gaussian_table = GenerateGaussianTable(config.color_sigma, 256);

	for (int y = r; y < h-r; ++y) {
		for (int x = r; x < w-r; ++x) {
			const uint8_t *base_in = &in[w*y+x];
			uint8_t *base_out = &out[w*y+x];
			float weight_sum = 0.0f;
			float weight_pixel_sum = 0.0f;
			for (int dy = -r; dy <= r; dy++) {
				for (int dx = -r; dx <= r; dx++) {
					int range_xdiff = abs(dx);
					int range_ydiff = abs(dy);
					int color_diff = abs(base_in[0] - base_in[dy*w+dx]);
					float weight =
						  color_gaussian_table.get()[color_diff]
						* range_gaussian_table.get()[range_xdiff]
						* range_gaussian_table.get()[range_ydiff];
					weight_sum += weight;
					weight_pixel_sum += weight * base_in[dy*w+dx];
				}
			}
			base_out[0] = ClampToUint8(int(weight_pixel_sum/weight_sum + 0.5f));
		}
	}
}

inline int CeilDiv(const int a, const int b)
{
	return (a-1)/b+1;
}

void bilateral_ocl(const uint8_t *in, uint8_t *out, const BilateralConfig config, const int w, const int h)
{
	const int r = config.radius;
	if (w <= 2*r || h <= 2*r) {
		LOG(WARNING) << "No work to do";
		return;
	}
	auto range_gaussian_table = GenerateGaussianTable(config.range_sigma, r+1);
	auto color_gaussian_table = GenerateGaussianTable(config.color_sigma, 256);
	cl_kernel kernel = device_manager->GetKernel("bilateral.cl", "bilateral");

	auto d_range_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, (r+1)*sizeof(float));
	auto d_color_gaussian_table = device_manager->AllocateMemory(CL_MEM_READ_ONLY, 256*sizeof(float));
	auto d_in = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*sizeof(uint8_t));
	auto d_out = device_manager->AllocateMemory(CL_MEM_READ_WRITE, w*h*sizeof(uint8_t));
	device_manager->WriteMemory(range_gaussian_table.get(), *d_range_gaussian_table.get(), (r+1)*sizeof(float));
	device_manager->WriteMemory(color_gaussian_table.get(), *d_color_gaussian_table.get(), 256*sizeof(float));
	device_manager->WriteMemory(in, *d_in.get(), w*h*sizeof(uint8_t));

	const int work_w = w-2*r;
	const int work_h = h-2*r;
	const size_t block_dim[2] = { r, r };
	const size_t grid_dim[2] = { w, h };

	// LOG(INFO) << "in opencl:\n";

	/*TODO: call the kernel*/
	vector<pair<const void*, size_t>> arg_and_sizes;
	arg_and_sizes.push_back( pair<const void*, size_t>( d_in.get(), sizeof(cl_mem) ) );
	arg_and_sizes.push_back( pair<const void*, size_t>( d_out.get(), sizeof(cl_mem) ) );
	arg_and_sizes.push_back( pair<const void*, size_t>( &r, sizeof(int) ) );
	arg_and_sizes.push_back( pair<const void*, size_t>( &work_w, sizeof(int) ) );
	arg_and_sizes.push_back( pair<const void*, size_t>( &work_h, sizeof(int) ) );
	arg_and_sizes.push_back( pair<const void*, size_t>( &w, sizeof(int) ) );
	arg_and_sizes.push_back( pair<const void*, size_t>( d_range_gaussian_table.get(), sizeof(cl_mem) ) );
	arg_and_sizes.push_back( pair<const void*, size_t>( d_color_gaussian_table.get(), sizeof(cl_mem) ) );

	device_manager->Call( kernel, arg_and_sizes, 2, grid_dim, NULL, block_dim );

	device_manager->ReadMemory(out, *d_out.get(), w*h*sizeof(uint8_t));
}


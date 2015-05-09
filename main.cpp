#include "cl_helper.h"
#include "pgm.h"
#include "global.h"
#include "bilateral.h"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <CL/cl.h>
#include <chrono>
#include <cmath>
#include <cstdint>
using namespace std;
using namespace google;
using namespace std::chrono;
DeviceManager *device_manager;

DEFINE_uint64(radius, 5, "Bilateral window radius");
DEFINE_uint64(device, 0, "Device to use");
DEFINE_double(r_sigma, 10, "Bilateral filter range sigma");
DEFINE_double(c_sigma, 30, "Bilateral filter color sigma");
static bool ValidDouble(const char *name, double d)
{
	bool valid = d > 0;
	LOG_IF(INFO, !valid) << name << " " << d << " must > 0";
	return valid;
}

void InitOpenCL(size_t id)
{
	// platforms
	auto platforms = GetPlatforms();
	LOG(INFO) << platforms.size() << " platform(s) found";
	int last_nvidia_platform = -1;
	for (size_t i = 0; i < platforms.size(); ++i) {
		auto platform_name = GetPlatformName(platforms[i]);
		LOG(INFO) << ">>> Name: " << platform_name.data();
		if (strcmp("NVIDIA CUDA", platform_name.data()) == 0) {
			last_nvidia_platform = i;
		}
	}
	CHECK_NE(last_nvidia_platform, -1) << "Cannot find any NVIDIA CUDA platform";

	// devices under the last CUDA platform
	auto devices = GetPlatformDevices(platforms[last_nvidia_platform]);
	LOG(INFO) << devices.size() << " device(s) found under some platform";
	for (size_t i = 0; i < devices.size(); ++i) {
		auto device_name = GetDeviceName(devices[i]);
		LOG(INFO) << ">>> Name: " << device_name.data();
	}
	CHECK_LT(id, devices.size()) << "Cannot find device " << id;
	device_manager = new DeviceManager(devices[id]);
}

int main(int argc, char** argv)
{
	InitGoogleLogging(argv[0]);
	FLAGS_logtostderr = true;
	RegisterFlagValidator(&FLAGS_c_sigma, &ValidDouble);
	RegisterFlagValidator(&FLAGS_r_sigma, &ValidDouble);
	SetUsageMessage("Usage: executable [gflag options] <input_file>");
	ParseCommandLineFlags(&argc, &argv, true);
	CHECK_GT(argc, 1) << "You must provide input file";
	char *input_file = argv[1];
	InitOpenCL(FLAGS_device);

	int w, h, c;
	const int r = FLAGS_radius;
	const float color_sigma = FLAGS_c_sigma;
	const float range_sigma = FLAGS_r_sigma;
	bool success;

	// Read and allocate the images
	unique_ptr<uint8_t[]> image_in = ReadNetpbm(w, h, c, success, input_file);
	CHECK(success) << "Cannot load image";
	LOG(INFO) << "Load " << w << 'x' << h << 'x' << c << " image";
	CHECK_EQ(c, 1) << "Please use a grayscale image";
	unique_ptr<uint8_t[]> image_out_gold(new uint8_t[w*h*c]);
	unique_ptr<uint8_t[]> image_out_your(new uint8_t[w*h*c]);

	for( size_t i = 0; i < w*h*c; ++i ){
		image_out_your.get()[i] = 0;
	}

	// Let's run the code
	time_point<high_resolution_clock> tic, toc;
	microseconds::rep elapsed_cxx, elapsed_ocl;

	// C++
	tic = high_resolution_clock::now();
	bilateral_cxx(image_in.get(), image_out_gold.get(), {r, range_sigma, color_sigma}, w, h);
	toc = high_resolution_clock::now();
	elapsed_cxx = duration_cast<microseconds>(toc-tic).count();

	// OpenCL
	device_manager->GetKernel("bilateral.cl", "bilateral"); // preload the kernel
	tic = high_resolution_clock::now();
	bilateral_ocl(image_in.get(), image_out_your.get(), {r, range_sigma, color_sigma}, w, h);
	toc = high_resolution_clock::now();
	elapsed_ocl = duration_cast<microseconds>(toc-tic).count();

	LOG(INFO) << "Without OpenCL: " << elapsed_cxx << "ms";
	LOG(INFO) << "With OpenCL: " << elapsed_ocl << "ms";

	// Output and check the result
	WritePGM(image_out_your.get(), w, h, "your.pgm");
	WritePGM(image_out_gold.get(), w, h, "gold.pgm");
	int warning_count = 0;
	int error_count = 0;
	for (int y = r; y < h-r; ++y) {
		for (int x = r; x < w-r; ++x) {
			int diff = abs(image_out_gold.get()[w*y+x] - image_out_your.get()[w*y+x]);
			if (diff >= 4) {
				error_count++;
			} else if (diff >= 2){
				warning_count++;
			}
		}
	}
	LOG_IF(WARNING, warning_count) << warning_count << " warning pixels";
	LOG_IF(ERROR, error_count) << error_count << " error pixels";

	delete device_manager;
	return 0;
}

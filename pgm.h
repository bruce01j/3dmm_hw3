#pragma once
#include <glog/logging.h>
#include <memory>
using namespace google;
using std::unique_ptr;

unique_ptr<uint8_t[]> ReadNetpbm(int &width, int &height, int &num_channel, bool &success, const char *filename);

template <class T>
void WriteNetpbm(T* i, const int width, const int height, const int num_channel, const char *filename, const char *magic)
{
	const int num_pixel = width*height;
	const int num_element = num_pixel*num_channel;
	unique_ptr<uint8_t[]> buffer(new uint8_t[num_element]);
	copy(i, i+num_element, buffer.get());
	FILE *fp = fopen(filename, "wb");
	CHECK_NOTNULL(fp);
	fprintf(fp, "%s\n%d %d\n255\n", magic, width, height);
	fwrite(buffer.get(), 1, num_element, fp);
}

template <class T>
void WritePGM(T* i, const int width, const int height, const char *filename)
{
	WriteNetpbm(i, width, height, 1, filename, "P5");
}

template <class T>
void WritePPM(T* i, const int width, const int height, const char *filename)
{
	WriteNetpbm(i, width, height, 3, filename, "P6");
}

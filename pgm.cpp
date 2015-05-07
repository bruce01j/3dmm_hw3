#include "pgm.h"
#include <algorithm>
using std::transform;

unique_ptr<uint8_t[]> ReadNetpbm(int &width, int &height, int &num_channel, bool &success, const char *filename)
{
	unique_ptr<uint8_t[]> buffer;
	success = false;

	do {
		char magic[3];
		int max_value, num_read;
		FILE *fp = fopen(filename, "rb");
		CHECK_NOTNULL(fp);

		// P5, P6
		num_read = fread(magic, 1, 3, fp);
		if (num_read != 3 || magic[0] != 'P' || magic[2] != '\n') {
			break;
		} else if (magic[1] == '5') {
			num_channel = 1;
		} else if (magic[1] == '6') {
			num_channel = 3;
		} else {
			break;
		}

		// comment
		while(true) {
			int peek_character = fgetc(fp);
			if (peek_character == '#') {
				while(fgetc(fp) != '\n') {}
			} else {
				ungetc(peek_character, fp);
				break;
			}
		}

		// image dimension
		num_read = fscanf(fp, "%d %d\n%d\n", &width, &height, &max_value);
		if (num_read != 3) {
			DLOG(INFO) << "Invalid width/height format";
			break;
		}

		// image data
		const int num_pixel = width*height;
		const int num_element = num_pixel*num_channel;
		buffer.reset(new uint8_t [num_element]);
		num_read = fread(buffer.get(), 1, num_element, fp);
		if (num_read != num_element) {
			DLOG(INFO) << "Wrong pixel number";
			break;
		}

		// normalize if required
		if (max_value != 255) {
			transform(buffer.get(), buffer.get()+num_element, buffer.get(), [max_value](const uint8_t in) -> uint8_t {
				return (255*in+max_value/2)/max_value;
			});
		}
		success = true;
	} while (false);

	return move(buffer);
}

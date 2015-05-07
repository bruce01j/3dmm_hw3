__kernel void bilateral(
		__global const uchar *in,
		__global uchar *out,
		const int r,
		const int work_w,
		const int work_h,
		const int row_stride,
		__constant float *range_gaussian_table,
		__constant float *color_gaussian_table
) {
}

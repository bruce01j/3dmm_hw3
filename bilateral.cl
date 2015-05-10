__kernel void bilateral(
		__global const uchar *in,
		__global uchar *out,
		const int r,
		const int work_w,
		const int work_h,
		const int row_stride,
		__constant float *range_gaussian_table,
		__constant float *color_gaussian_table
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int id = y * row_stride + x;

    if( x < r || x >= work_w + r || y < r || y >= work_h + r ){
    }
    else{
        float weight_sum = 0.0f;
        float weight_pixel_sum = 0.0f;
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                int range_xdiff = abs(dx);
                int range_ydiff = abs(dy);
                int color_diff = abs(in[id] - in[id+dy*row_stride+dx]);
                float weight =
                      color_gaussian_table[color_diff]
                    * range_gaussian_table[range_xdiff]
                    * range_gaussian_table[range_ydiff];
                weight_sum += weight;
                weight_pixel_sum += weight * in[id+dy*row_stride+dx];
            }
        }
        out[id] = convert_uchar(weight_pixel_sum/weight_sum + 0.5f);
    }
}


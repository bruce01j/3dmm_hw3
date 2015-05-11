__kernel void bilateral(
		__global const uchar *in,
		__global uchar *out,
		const int r,
		const int work_w,
		const int work_h,
		const int row_stride,
		__constant float *range_gaussian_table,
		__constant float *color_gaussian_table,
		__local float *local_range_gaussian_table,
		__local float *local_color_gaussian_table
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int id = ( y + r ) * row_stride + x + r;
    int lw = get_local_size(0);
    int lh = get_local_size(1);
    int local_stride = 2 * r + lw;
    int lid = ( ly + r ) * local_stride + lx + r;

    for( int j = ly; j*lw < 256; j += lh ){
        for( int i = lx; j*lw + i <  256; i += lw ){
            local_color_gaussian_table[i+j*lw] = color_gaussian_table[i+j*lw];
        }
    }
    for( int j = ly; j*lw < r+1; j += lh ){
        for( int i = lx; j*lw + i <  r+1; i += lw ){
            local_range_gaussian_table[i+j*lw] = range_gaussian_table[i+j*lw];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if( x < work_w || y < work_h ){
        float weight_sum = 0.0f;
        float weight_pixel_sum = 0.0f;
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                int range_xdiff = abs(dx);
                int range_ydiff = abs(dy);
                uchar dest_c = in[id+dy*row_stride+dx];
                int color_diff = abs(in[id] - dest_c);
                float weight =
                      local_color_gaussian_table[color_diff]
                    * local_range_gaussian_table[range_xdiff]
                    * local_range_gaussian_table[range_ydiff];
                weight_sum += weight;
                weight_pixel_sum += weight * dest_c;
            }
        }
        out[id] = convert_uchar(weight_pixel_sum/weight_sum + 0.5f);
    }
}

__kernel void bilateral(
		__global const uchar *in,
		__global uchar *out,
		const int r,
		const int work_w,
		const int work_h,
		const int row_stride,
		__constant float *range_gaussian_table,
		__constant float *color_gaussian_table,
        __local uchar* patch
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

    for( int i = 0; i < r+r+lw-lx; i += lw ){
        for( int j = 0; j < r+r+lh-ly; j += lh ){
            patch[i+lx+(j+ly)*local_stride] = in[x+i+(j+y)*row_stride];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if( x >= work_w || y >= work_h ){
    }
    else{
        float weight_sum = 0.0f;
        float weight_pixel_sum = 0.0f;
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                int range_xdiff = abs(dx);
                int range_ydiff = abs(dy);
                int color_diff = abs(patch[lid] - patch[lid+dy*local_stride+dx]);
                float weight =
                      color_gaussian_table[color_diff]
                    * range_gaussian_table[range_xdiff]
                    * range_gaussian_table[range_ydiff];
                weight_sum += weight;
                weight_pixel_sum += weight * patch[lid+dy*local_stride+dx];
            }
        }
        out[id] = convert_uchar(weight_pixel_sum/weight_sum + 0.5f);
    }
}

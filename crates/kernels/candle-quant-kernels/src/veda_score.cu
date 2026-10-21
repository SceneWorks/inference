// Frozen Veda tile scorer (gen-core `contract::pool_video_tiles` + `contract::tile_logits`) on the
// device, bit for bit. One thread computes one output with the host's exact operation order:
// sequential FP32 sums from 0.0f, each product rounded before its add. `__fmul_rn`/`__fadd_rn`/
// `__fdiv_rn` are never contracted into FMA or reassociated, so an output depends only on its own
// operands and never on how many heads, tiles or threads share the launch.

// x: [heads, tiles, 128, dim] FP32, padding rows zero. f: [heads, video, 3*dim] = mean|max|min
// over each video tile's `valid` real rows; the mean sums all 128 rows in row order, as the host.
extern "C" __global__ void veda_pool(const float* x, const unsigned int* valid, float* f,
    int heads, int tiles, int video, int dim) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)heads*video*dim) return;
    int d=(int)(i%dim), t=(int)((i/dim)%video), h=(int)(i/((long long)dim*video));
    unsigned int count=valid[t];
    long long out=((long long)h*video+t)*3*dim+d;
    if(count==0) { f[out]=0.0f; f[out+dim]=0.0f; f[out+2*dim]=0.0f; return; }
    const float pos=__int_as_float(0x7f800000);
    float sum=0.0f, mx=-pos, mn=pos;
    const float* row=x+(((long long)h*tiles+t)*128)*dim+d;
    for(int r=0;r<128;r++) {
        float v=row[(long long)r*dim];
        sum=__fadd_rn(sum,v);
        if((unsigned int)r<count) { mx=fmaxf(mx,v); mn=fminf(mn,v); }
    }
    f[out]=__fdiv_rn(sum,(float)count);
    f[out+dim]=mx;
    f[out+2*dim]=mn;
}
// f: [heads, video, 3*dim]; p: [heads, 3*dim, dim]. out[h,t,d] = (sum_i f[h,t,i]*p[h,i,d]) + f[h,t,d].
extern "C" __global__ void veda_project(const float* f, const float* p, float* out,
    int heads, int video, int dim) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)heads*video*dim) return;
    int d=(int)(i%dim), t=(int)((i/dim)%video), h=(int)(i/((long long)dim*video));
    const float* fr=f+((long long)h*video+t)*3*dim;
    const float* pr=p+(long long)h*3*dim*dim+d;
    float v=0.0f;
    for(int k=0;k<3*dim;k++) v=__fadd_rn(v,__fmul_rn(fr[k],pr[(long long)k*dim]));
    out[i]=__fadd_rn(v,fr[d]);
}
// q,k: [heads, video, dim]. s[h,r,c] = (sum_d q[h,r,d]*k[h,c,d]) / divisor.
extern "C" __global__ void veda_logits(const float* q, const float* k, float* s,
    int heads, int video, int dim, float divisor) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)heads*video*video) return;
    int c=(int)(i%video), r=(int)((i/video)%video), h=(int)(i/((long long)video*video));
    const float* qr=q+((long long)h*video+r)*dim;
    const float* kr=k+((long long)h*video+c)*dim;
    float v=0.0f;
    for(int d=0;d<dim;d++) v=__fadd_rn(v,__fmul_rn(qr[d],kr[d]));
    s[i]=__fdiv_rn(v,divisor);
}

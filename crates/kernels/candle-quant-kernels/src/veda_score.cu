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
// Source coefficients round E4M3 * per-head F32 scale to BF16 before the frozen FP32 dot.
__device__ __forceinline__ float veda_coeff(const float* values,const float*,long long i,int) {
    return values[i];
}
__device__ __forceinline__ float veda_coeff(const unsigned char* codes,const float* scales,long long i,int head) {
    unsigned char b=codes[i];
    int exp=(b>>3)&15, mant=b&7;
    float value=exp?ldexpf(1.0f+mant*0.125f,exp-7):mant*0.001953125f;
    float scaled=__fmul_rn((b&128)?-value:value,scales[head]);
    unsigned int bits=__float_as_uint(scaled);
    return __uint_as_float((bits+0x7fff+((bits>>16)&1))&0xffff0000);
}
template<typename T>
__device__ __forceinline__ void veda_project_impl(const float* f,const T* p,const float* scales,float* out,int heads,int video,int dim) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)heads*video*dim) return;
    int d=(int)(i%dim), t=(int)((i/dim)%video), h=(int)(i/((long long)dim*video));
    const float* fr=f+((long long)h*video+t)*3*dim;
    long long offset=(long long)h*3*dim*dim+d;
    float v=0.0f;
    for(int k=0;k<3*dim;k++) v=__fadd_rn(v,__fmul_rn(fr[k],veda_coeff(p,scales,offset+(long long)k*dim,h)));
    out[i]=__fadd_rn(v,fr[d]);
}
extern "C" __global__ void veda_project(const float* f,const float* p,float* out,int heads,int video,int dim) {
    veda_project_impl(f,p,nullptr,out,heads,video,dim);
}
extern "C" __global__ void veda_project_fp8(const float* f,const unsigned char* codes,const float* scales,float* out,int heads,int video,int dim) {
    veda_project_impl(f,codes,scales,out,heads,video,dim);
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

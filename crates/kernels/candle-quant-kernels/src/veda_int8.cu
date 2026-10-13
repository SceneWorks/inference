// Native pointer port of Veda-on-ComfyUI@60bfae6 sparse_int8.py.
// MIT Veda notice: veda-LICENSE.upstream. Sage arithmetic attribution: veda-LICENSE.sage.
// One CTA per real/padded query row; walks only its retained 64-row key blocks.
// No TMA, cluster, alternate sparse route, or quadratic score workspace.
__device__ float half_round(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    float y;
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
    return y;
}
extern "C" __global__ void veda_quant(const float* x, signed char* out,
    float* scales, int slots, int dim, int block, float pre) {
    __shared__ float peak[128];
    int tid=threadIdx.x, head=blockIdx.y, bi=blockIdx.x;
    long long base=((long long)head*slots+bi*block)*dim;
    float a=0;
    for(int i=tid;i<block*dim;i+=128) a=fmaxf(a,fabsf(x[base+i]*pre));
    peak[tid]=a; __syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(tid<stride) peak[tid]=fmaxf(peak[tid],peak[tid+stride]);
        __syncthreads();
    }
    float scale=peak[0]>0?peak[0]/127.0f:1.0f;
    if(tid==0) scales[head*(slots/block)+bi]=scale;
    for(int i=tid;i<block*dim;i+=128) {
        float z=x[base+i]*pre/scale;
        out[base+i]=(signed char)(int)(z+(z>=0?0.5f:-0.5f));
    }
}
extern "C" __global__ void veda_attend(const signed char* q,const signed char* k,
    const float* v,const float* qs,const float* ks,const unsigned int* offsets,
    const unsigned int* indices,const unsigned int* valid,float* out,int slots,int dim) {
    __shared__ float logits[128];
    __shared__ float probabilities[64];
    __shared__ float row_logits[64];
    int tid=threadIdx.x,row=blockIdx.x,head=blockIdx.y,tile=row/128;
    int nt=slots/128, nb=slots/64;
    unsigned int begin=offsets[head*nt+tile],end=offsets[head*nt+tile+1];
    long long qb=((long long)head*slots+row)*dim;
    float m=-1e30f,l=0,acc=0;
    for(unsigned int at=begin;at<end;at++) {
        unsigned int block=indices[at];
        float dot=-__int_as_float(0x7f800000);
        if(tid<64 && tid<valid[block]) {
            long long kb=((long long)head*slots+block*64+tid)*dim;
            int sum=0;
            // Four signed bytes per integer dot instruction. Width is validated a multiple of 4.
            for(int d=0;d<dim;d+=4) {
                int aq=*(const int*)(q+qb+d),ak=*(const int*)(k+kb+d);
                asm("dp4a.s32.s32 %0, %1, %2, %3;":"=r"(sum):"r"(aq),"r"(ak),"r"(sum));
            }
            dot=(float)sum*qs[head*nt+tile]*ks[head*nb+block];
        }
        if(tid<64) row_logits[tid]=dot;
        logits[tid]=dot; __syncthreads();
        for(int stride=64;stride;stride/=2) {
            if(tid<stride) logits[tid]=fmaxf(logits[tid],logits[tid+stride]);
            __syncthreads();
        }
        float next=fmaxf(m,logits[0]), alpha=exp2f(m-next);
        if(tid<64) {
            float z=(tid<valid[block])?exp2f(row_logits[tid]-next):0;
            probabilities[tid]=half_round(z); logits[tid]=z;
        } else logits[tid]=0;
        __syncthreads();
        for(int stride=64;stride;stride/=2) {
            if(tid<stride) logits[tid]+=logits[tid+stride];
            __syncthreads();
        }
        l=l*alpha+logits[0];
        if(tid<dim) {
            float pv=0;
            for(int n=0;n<64;n++) {
                float val=half_round(v[((long long)head*slots+block*64+n)*dim+tid]);
                pv=fmaf(probabilities[n],val,pv);
            }
            acc=acc*alpha+half_round(pv);
        }
        m=next; __syncthreads();
    }
    // Padding queries are defined zero; production inverse permutation discards them.
    if(tid<dim) out[qb+tid]=(row%128<valid[tile*2]+valid[tile*2+1])?acc/(l>0?l:1):0;
}

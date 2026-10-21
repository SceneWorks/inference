// Native pointer port of Veda-on-ComfyUI@60bfae6 sparse_int8.py.
// MIT Veda notice: veda-LICENSE.upstream. Sage arithmetic attribution: veda-LICENSE.sage.
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
// One 128-thread CTA per (32 query rows, head). Every row of a 128-row query tile retains the same
// 64-row key blocks, so each retained block's INT8 K and FP16-rounded V are staged in shared memory
// once for all 32 rows instead of being re-read from global memory by one CTA per row. Lane r owns
// query row r of the group; warp w scores keys 16w..16w+15 of the block and owns output dims
// 32w..32w+31. Per row the arithmetic is the frozen per-row recipe, operation for operation: exact
// DP4A integer dots scaled by the Q tile and K block scales, block max, exp2 against the running
// max, an FP32 sum of the unrounded exponentials in the same pairwise order (stride 32..1), FP16
// probabilities and V, a sequential FP32 FMA over the block's 64 keys, an FP16-rounded block PV and
// the FP32 rescale. Invalid keys score -inf and contribute probability 0.
#define VEDA_ROWS 32
#define VEDA_QSTRIDE 132  // 33 words: the 32 lanes read 32 distinct banks
extern "C" __global__ void __launch_bounds__(128) veda_attend(const signed char* q,
    const signed char* k, const float* v, const float* qs, const float* ks,
    const unsigned int* offsets, const unsigned int* indices, const unsigned int* valid,
    float* out, int slots, int dim) {
    __shared__ __align__(16) signed char qsm[VEDA_ROWS*VEDA_QSTRIDE];
    // K block as [64][128] bytes while scoring, then the block's logits as [64 keys][32 rows].
    __shared__ __align__(16) float kl[64*VEDA_ROWS];
    __shared__ __align__(16) float vsm[64*128];
    signed char* ksm=(signed char*)kl;
    const float inf=__int_as_float(0x7f800000);
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5, head=blockIdx.y;
    int row0=blockIdx.x*VEDA_ROWS, row=row0+lane, tile=row0/128;
    int nt=slots/128, nb=slots/64, words=dim/4;
    long long hb=(long long)head*slots;
    unsigned int begin=offsets[head*nt+tile],end=offsets[head*nt+tile+1];
    for(int i=tid;i<VEDA_ROWS*words;i+=128) {
        int r=i/words,c=i-r*words;
        *(int*)(qsm+r*VEDA_QSTRIDE+4*c)=*(const int*)(q+(hb+row0+r)*dim+4*c);
    }
    float qscale=qs[head*nt+tile];
    float m=-1e30f,l=0;
    float acc[32];
    #pragma unroll
    for(int j=0;j<32;j++) acc[j]=0;
    for(unsigned int at=begin;at<end;at++) {
        unsigned int block=indices[at];
        int nvalid=(int)valid[block];
        __syncthreads();  // Q is staged and the previous block's readers of kl/vsm are done
        long long kb=hb+(long long)block*64;
        for(int i=tid;i<64*words;i+=128) {
            int r=i/words,c=i-r*words;
            *(int*)(ksm+r*128+4*c)=*(const int*)(k+(kb+r)*dim+4*c);
        }
        for(int i=tid;i<64*dim;i+=128) {
            int r=i/dim,c=i-r*dim;
            vsm[r*128+c]=half_round(v[(kb+r)*dim+c]);
        }
        __syncthreads();
        float kscale=ks[head*nb+block];
        int sums[16];
        #pragma unroll
        for(int i=0;i<16;i++) sums[i]=0;
        // Four signed bytes per integer dot instruction. Width is validated a multiple of 4.
        for(int d=0;d<words;d++) {
            int aq=*(const int*)(qsm+lane*VEDA_QSTRIDE+4*d);
            #pragma unroll
            for(int i=0;i<16;i++) {
                int ak=*(const int*)(ksm+(warp*16+i)*128+4*d);
                asm("dp4a.s32.s32 %0, %1, %2, %3;":"=r"(sums[i]):"r"(aq),"r"(ak),"r"(sums[i]));
            }
        }
        float dots[16];
        #pragma unroll
        for(int i=0;i<16;i++)
            dots[i]=(warp*16+i<nvalid)?(float)sums[i]*qscale*kscale:-inf;
        __syncthreads();  // every warp finished reading K before the logits overwrite it
        #pragma unroll
        for(int i=0;i<16;i++) kl[(warp*16+i)*VEDA_ROWS+lane]=dots[i];
        __syncthreads();
        float block_max=-inf;
        #pragma unroll
        for(int n=0;n<64;n++) block_max=fmaxf(block_max,kl[n*VEDA_ROWS+lane]);
        float next=fmaxf(m,block_max), alpha=exp2f(m-next);
        float p[64];
        #pragma unroll
        for(int n=0;n<64;n++) p[n]=(n<nvalid)?exp2f(kl[n*VEDA_ROWS+lane]-next):0;
        float s[32];
        #pragma unroll
        for(int i=0;i<32;i++) s[i]=p[i]+p[i+32];
        #pragma unroll
        for(int i=0;i<16;i++) s[i]+=s[i+16];
        #pragma unroll
        for(int i=0;i<8;i++) s[i]+=s[i+8];
        #pragma unroll
        for(int i=0;i<4;i++) s[i]+=s[i+4];
        s[0]+=s[2]; s[1]+=s[3];
        s[0]+=s[1];
        l=l*alpha+s[0];
        #pragma unroll
        for(int n=0;n<64;n++) p[n]=half_round(p[n]);
        float pv[32];
        #pragma unroll
        for(int j=0;j<32;j++) pv[j]=0;
        if(warp*32<dim) {
            #pragma unroll
            for(int n=0;n<64;n++) {
                const float4* vr=(const float4*)(vsm+n*128+warp*32);
                #pragma unroll
                for(int j=0;j<8;j++) {
                    float4 x=vr[j];
                    pv[4*j]=fmaf(p[n],x.x,pv[4*j]);
                    pv[4*j+1]=fmaf(p[n],x.y,pv[4*j+1]);
                    pv[4*j+2]=fmaf(p[n],x.z,pv[4*j+2]);
                    pv[4*j+3]=fmaf(p[n],x.w,pv[4*j+3]);
                }
            }
        }
        #pragma unroll
        for(int j=0;j<32;j++) acc[j]=acc[j]*alpha+half_round(pv[j]);
        m=next;
    }
    // Padding queries are defined zero; production inverse permutation discards them.
    bool live=(row%128)<(int)(valid[tile*2]+valid[tile*2+1]);
    #pragma unroll
    for(int j=0;j<32;j++) {
        int d=warp*32+j;
        if(d<dim) out[(hb+row)*dim+d]=live?acc[j]/(l>0?l:1):0;
    }
}

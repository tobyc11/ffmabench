#include <metal_stdlib>
using namespace metal;

float4 meaningful(float4 x, float4 y)
{
    float4 r1 = x;
    float4 r2 = y;
    for (int i = 0; i < 1000; i++)
    {
        r1 = fma(r1, x, r1); // 8 FLOP
        r2 = fma(r2, y, r2); // 8 FLOP
    }
    return r1 + r2;
}

[[kernel]] void bench_kernel(texture2d<float> tex [[texture(0)]],
                              sampler samp [[sampler(0)]],
                              device float4 *out [[buffer(0)]],
                              uint2 tid [[thread_position_in_threadgroup]])
{
    float4 c0 = { 1.0f, 2.0f, 3.0f, 4.0f };
    float4 c1 = { 1.0f, 2.1f, 3.2f, 4.3f };
    // float4 v0 = tex.sample(samp, float2(tid.x, tid.y));
    float4 v0 = float4(tid.x, tid.y, tid.x, tid.y);
    float4 v1 = v0 * c0 - c1;
    *out = meaningful(v0, v1);
}

[[kernel]] void bench2_kernel(texture2d<float> tex [[texture(0)]],
                             sampler samp [[sampler(0)]],
                             device float4 *out [[buffer(0)]],
                             uint2 tid [[thread_position_in_threadgroup]])
{
    float4 c0 = { 1.0f, 2.0f, 3.0f, 4.0f };
    float4 c1 = { 1.0f, 2.1f, 3.2f, 4.3f };
    float4 v0 = tex.sample(samp, float2(tid.x, tid.y));
    float4 v1 = v0;
    float4 v2 = v1 * v1;
    for (int i = 0; i < 1000; i++)
    {
        v1 = v1 * v0 + c1;
    }
    v2 = v1;
    for (int i = 0; i < 1000; i++)
    {
        v2 = v2 * v0 - c0;
    }
    *out = v2;
}

struct RastData
{
    float4 position [[position]];
    float2 uv;
};

vertex RastData test_vs(uint vid [[vertex_id]])
{
    RastData out;
    if (vid == 0) {
        out.position = float4(-1, -1, 0, 1);
        out.uv = float2(0, 0);
    } else if (vid == 1) {
        out.position = float4(1, -1, 0, 1);
        out.uv = float2(1, 0);
    } else if (vid == 2) {
        out.position = float4(1, 1, 0, 1);
        out.uv = float2(1, 1);
    } else if (vid == 3) {
        out.position = float4(-1, -1, 0, 1);
        out.uv = float2(0, 0);
    } else if (vid == 4) {
        out.position = float4(1, 1, 0, 1);
        out.uv = float2(1, 1);
    } else if (vid == 5) {
        out.position = float4(-1, 1, 0, 1);
        out.uv = float2(0, 1);
    }
    return out;
}

fragment float4 test_ps(RastData in [[stage_in]])
{
    float4 m = meaningful(in.uv.x, in.uv.y);
    m = meaningful(m, m + m);
    m = meaningful(m, m + m);
    m = meaningful(m, m + m);
    m = meaningful(m, m + m);
    m = clamp(m, float4(0), float4(1));
    return float4(in.position.x / 4096, in.uv.y, m.x + m.y + m.z + m.a, 1);
}

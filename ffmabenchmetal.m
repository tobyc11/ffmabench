#import <Metal/Metal.h>
#include <time.h>

static void* generate_arr_data()
{
    float *buffer = (float *)malloc(sizeof(float) * 4 * 1024);
    for (int i = 0; i < 1024; i++)
    {
        buffer[i * 4 + 0] = 1.0f;
        buffer[i * 4 + 1] = 2.0f;
        buffer[i * 4 + 2] = 3.01f;
        buffer[i * 4 + 3] = 4.01f;
    }
    return buffer;
}

static void write_to_ppm(unsigned char *buffer, int width, int height)
{
    const char *home = getenv("HOME");
    char fn[256];
    sprintf(fn, "%s/sample.ppm", home);
    printf("Writing into %s\n", fn);
    FILE *f = fopen(fn, "w");
    if (!f) {
        return;
    }
    
    fprintf(f, "P3\n");
    fprintf(f, "%d %d\n", width, height);
    fprintf(f, "255\n");
    
    unsigned int x, y;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (x != 0) {
                fprintf(f, " ");
            }
            unsigned int i = y * width + x;
            unsigned char *p = buffer + 4 * i;
            fprintf(f, "%u %u %u", p[0], p[1], p[2]);
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
}

const int width = 4096;
const int height = width;

static void graphics_churn(id<MTLDevice> device, id<MTLTexture> tex_target, bool no_wait)
{
    // Create render pass
    MTLRenderPassDescriptor *rp_desc = [[MTLRenderPassDescriptor alloc] init];
    rp_desc.colorAttachments[0].texture = tex_target;
    rp_desc.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp_desc.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp_desc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    
    // Create command encoder
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cmd_buf renderCommandEncoderWithDescriptor:rp_desc];
    
    // Shader setup
    id<MTLLibrary> lib = [device newDefaultLibrary];
    id<MTLFunction> vs = [lib newFunctionWithName:@"test_vs"];
    id<MTLFunction> ps = [lib newFunctionWithName:@"test_ps"];
    MTLRenderPipelineDescriptor *pipe_desc = [[MTLRenderPipelineDescriptor alloc] init];
    pipe_desc.vertexFunction = vs;
    pipe_desc.fragmentFunction = ps;
    pipe_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:pipe_desc error:NULL];
    
    // Draw call
    [enc setRenderPipelineState:pso];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:3 vertexCount:3];
    [enc endEncoding];
    [cmd_buf commit];
    if (!no_wait) {
        [cmd_buf waitUntilCompleted];
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSError *errors;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == Nil)
        {
            NSArray<id<MTLDevice>> *dev_array = MTLCopyAllDevices();
            device = dev_array.firstObject;
        }
        
        // Create texture
        MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
        td.width = width;
        td.height = height;
        td.pixelFormat = MTLPixelFormatRGBA8Unorm;
        td.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> tex_target = [device newTextureWithDescriptor:td];
        
        printf("Benchmarking pixel shader\n");
        uint64_t begin = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        graphics_churn(device, tex_target, NO);
        uint64_t end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        printf("Finished in %f s\n", (double)(end - begin) / 1e9);
        double total_ops;
        total_ops = (double)width * height * 5 * 16000;
        printf("  %f TFLOPS\n", total_ops / (double)(end - begin) * 1e9 / 1e12);
        
        void *buffer = malloc(4 * width * height);
        [tex_target getBytes:buffer bytesPerRow:4 * width fromRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0];
        write_to_ppm(buffer, width, height);
        
        id<MTLBuffer> out_buffer = [device newBufferWithLength:0x1000 options:MTLResourceOptionCPUCacheModeDefault];
        
        MTLTextureDescriptor *tex_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:1024 height:1 mipmapped:NO];
        id<MTLTexture> tex = [device newTextureWithDescriptor:tex_desc];
        void *data = generate_arr_data();
        [tex replaceRegion:MTLRegionMake2D(0, 0, 1024, 1) mipmapLevel:0 withBytes:data bytesPerRow:sizeof(float) * 4 * 1024];
        free(data);
        
        MTLSamplerDescriptor *sampler_desc = [[MTLSamplerDescriptor alloc] init];
        id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:sampler_desc];
        
        id<MTLLibrary> lib = [device newDefaultLibrary];
        id<MTLFunction> func = [lib newFunctionWithName:@"bench_kernel"];
        id<MTLComputePipelineState> compute_pipe = [device newComputePipelineStateWithFunction:func error:&errors];
        
        printf("Benchmarking compute shader\n");
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];
        [command_encoder setComputePipelineState:compute_pipe];
        [command_encoder setBuffer:out_buffer offset:0 atIndex:0];
        [command_encoder setSamplerState:sampler atIndex:0];
        [command_encoder setTexture:tex atIndex:0];
        [command_encoder dispatchThreadgroups:MTLSizeMake(1048576, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [command_encoder endEncoding];
        begin = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        printf("Finished in %f s\n", (double)(end - begin) / 1e9);
        total_ops = (double)1048576 * 128 * 16000;
        printf("  %f TFLOPS\n", total_ops / (double)(end - begin) * 1e9 / 1e12);
    }
    return 0;
}

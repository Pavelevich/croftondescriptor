//
//  MetalCapabilities.mm
//  Crofton Descriptor - Metal Optimization
//

#import "MetalCapabilities.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static id<MTLDevice> _sharedDevice = nil;

@implementation MetalCapabilities

+ (void)initialize {
    if (self == [MetalCapabilities class]) {
        _sharedDevice = MTLCreateSystemDefaultDevice();
    }
}

+ (BOOL)isMetalAvailable {
    return _sharedDevice != nil;
}

+ (NSString*)metalDeviceName {
    return _sharedDevice ? [_sharedDevice name] : @"No Metal Device";
}

+ (BOOL)isAppleSilicon {
    if (!_sharedDevice) return NO;

    // Check for unified memory architecture (Apple Silicon feature)
    return _sharedDevice.hasUnifiedMemory;
}

+ (NSUInteger)maxThreadgroupMemory {
    if (!_sharedDevice) return 0;
    return _sharedDevice.maxThreadgroupMemoryLength;
}

+ (NSUInteger)recommendedThreadgroupSize {
    if (!_sharedDevice) return 64;

    // Apple Silicon optimal: 32-64 threads per group for compute
    if ([self isAppleSilicon]) {
        return 64;
    }

    // Intel/AMD discrete: 256-512
    return 256;
}

+ (MTLSize)recommendedThreadgroupSize2D {
    if ([self isAppleSilicon]) {
        // Apple Silicon: 8x8 or 16x16 optimal for image processing
        return MTLSizeMake(16, 16, 1);
    }

    // Discrete GPU: 16x16 or 32x32
    return MTLSizeMake(16, 16, 1);
}

+ (void)logCapabilities {
    if (!_sharedDevice) {
        NSLog(@"❌ Metal is not available on this device");
        return;
    }

    NSLog(@"");
    NSLog(@"╔════════════════════════════════════════════════════════════╗");
    NSLog(@"║          Metal Device Capabilities                         ║");
    NSLog(@"╠════════════════════════════════════════════════════════════╣");
    NSLog(@"║ Device Name:            %-34@║", [_sharedDevice name]);
    NSLog(@"║ Apple Silicon:          %-34@║", [self isAppleSilicon] ? @"YES ✅" : @"NO");
    NSLog(@"║ Unified Memory:         %-34@║", _sharedDevice.hasUnifiedMemory ? @"YES ✅" : @"NO");
    NSLog(@"║ Max Threadgroup Memory: %-30lu KB ║", [self maxThreadgroupMemory] / 1024);
    NSLog(@"║ Recommended TG Size:    %-34lu║", [self recommendedThreadgroupSize]);
    NSLog(@"║ Max Threads Per TG:     %-34lu║", _sharedDevice.maxThreadsPerThreadgroup.width);
    NSLog(@"║ Max Buffer Length:      %-28lu MB ║", _sharedDevice.maxBufferLength / (1024 * 1024));
    NSLog(@"║ Supports MPS:           %-34@║", @"YES ✅");

    MTLSize tg2D = [self recommendedThreadgroupSize2D];
    NSLog(@"║ Optimal 2D TG:          %lux%lu%-29@║", tg2D.width, tg2D.height, @"");
    NSLog(@"╚════════════════════════════════════════════════════════════╝");
    NSLog(@"");
}

+ (id<MTLDevice>)sharedDevice {
    return _sharedDevice;
}

+ (MTLResourceOptions)recommendedBufferStorageMode {
    if ([self isAppleSilicon]) {
        // Unified memory: use shared mode for zero-copy access
        return MTLResourceStorageModeShared;
    } else {
        // Discrete GPU: use managed mode for automatic synchronization
        return MTLResourceStorageModeManaged;
    }
}

@end

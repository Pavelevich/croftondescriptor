//
//  MetalCapabilities.h
//  Crofton Descriptor - Metal Optimization
//
//  Metal device capability detection and optimization recommendations
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface MetalCapabilities : NSObject

/// Check if Metal is available on this device
+ (BOOL)isMetalAvailable;

/// Get Metal device name
+ (NSString*)metalDeviceName;

/// Check if running on Apple Silicon (unified memory architecture)
+ (BOOL)isAppleSilicon;

/// Get maximum threadgroup memory available
+ (NSUInteger)maxThreadgroupMemory;

/// Get recommended threadgroup size for this device
+ (NSUInteger)recommendedThreadgroupSize;

/// Get optimal threadgroup dimensions for 2D image processing
+ (MTLSize)recommendedThreadgroupSize2D;

/// Log all device capabilities (for debugging)
+ (void)logCapabilities;

/// Get singleton Metal device
+ (id<MTLDevice>)sharedDevice;

/// Get recommended buffer storage mode for this device
+ (MTLResourceOptions)recommendedBufferStorageMode;

@end

NS_ASSUME_NONNULL_END

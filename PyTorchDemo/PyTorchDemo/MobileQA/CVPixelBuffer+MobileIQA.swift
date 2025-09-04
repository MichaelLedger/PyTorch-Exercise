import Accelerate
import Foundation
import UIKit

extension CVPixelBuffer {
    func normalizedForMobileIQA(_ width: Int, _ height: Int) -> [Float]? {
        let w = CVPixelBufferGetWidth(self)
        let h = CVPixelBufferGetHeight(self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(self)
        let bytesPerPixel = 4
        
        CVPixelBufferLockBaseAddress(self, .readOnly)
        guard let baseAddr = CVPixelBufferGetBaseAddress(self) else {
            return nil
        }
        
        // Create vImage buffers
        var inBuff = vImage_Buffer(data: baseAddr,
                                 height: UInt(h),
                                 width: UInt(w),
                                 rowBytes: bytesPerRow)
        
        guard let dstData = malloc(width * height * bytesPerPixel) else {
            return nil
        }
        
        var outBuff = vImage_Buffer(data: dstData,
                                  height: UInt(height),
                                  width: UInt(width),
                                  rowBytes: width * bytesPerPixel)
        
        // Try different resampling methods
        var flags: Int32 = 0
        switch TeacherModelConstants.resamplingMethod {
        case .nearest:
            flags = Int32(kvImageNoFlags)
        case .bilinear:
            flags = Int32(kvImageNoFlags | kvImageHighQualityResampling)
        case .highQuality:
            flags = Int32(kvImageHighQualityResampling)
        case .lanczos:
            // Lanczos-like high quality resampling with edge handling
            flags = Int32(kvImageHighQualityResampling | kvImageEdgeExtend)
        }
        let resamplingFlags = vImage_Flags(flags)
        
        let err = vImageScale_ARGB8888(&inBuff, &outBuff, nil, resamplingFlags)
        CVPixelBufferUnlockBaseAddress(self, .readOnly)
        
        if err != kvImageNoError {
            free(dstData)
            return nil
        }
        
        // Create normalized buffer with MobileIQA normalization parameters
        let batchSize = 1024 * 1024  // Process 1M pixels at a time
        var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: width * height * 3)
        let mean = TeacherModelConstants.normalizationMean
        let std = TeacherModelConstants.normalizationStd
        
        // Process in batches to reduce memory pressure
        for batchStart in stride(from: 0, to: width * height, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, width * height)
            
            // Process each batch
            for i in batchStart ..< batchEnd {
                // Get raw RGB values
                let r = Float32(dstData.load(fromByteOffset: i * 4 + 2, as: UInt8.self))
                let g = Float32(dstData.load(fromByteOffset: i * 4 + 1, as: UInt8.self))
                let b = Float32(dstData.load(fromByteOffset: i * 4 + 0, as: UInt8.self))
                
                // Print first few pixels for debugging
                if i < 5 {
                    print("Pixel \(i): R=\(r), G=\(g), B=\(b)")
                }
                
                // Normalize values
                normalizedBuffer[i] = (r / 255.0 - mean[0]) / std[0] // R
                normalizedBuffer[width * height + i] = (g / 255.0 - mean[1]) / std[1] // G
                normalizedBuffer[width * height * 2 + i] = (b / 255.0 - mean[2]) / std[2] // B
                
                // Print first few normalized values for debugging
                if i < 5 {
                    print("Normalized \(i): R=\(normalizedBuffer[i]), G=\(normalizedBuffer[width * height + i]), B=\(normalizedBuffer[width * height * 2 + i])")
                }
            }
            
            // Optional: Release memory pressure
            autoreleasepool { }
        }
        
        free(dstData)
        return normalizedBuffer
    }
}

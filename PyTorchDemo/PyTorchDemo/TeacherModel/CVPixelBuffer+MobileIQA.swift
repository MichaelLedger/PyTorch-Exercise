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
        
        // Scale the image using high-quality resampling
        let err = vImageScale_ARGB8888(&inBuff, &outBuff, nil, vImage_Flags(kvImageHighQualityResampling))
        CVPixelBufferUnlockBaseAddress(self, .readOnly)
        
        if err != kvImageNoError {
            free(dstData)
            return nil
        }
        
        // Create normalized buffer with MobileIQA normalization parameters
        var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: width * height * 3)
        let mean = TeacherModelConstants.normalizationMean
        let std = TeacherModelConstants.normalizationStd
        
        // Normalize pixel values using MobileIQA parameters
        for i in 0 ..< width * height {
            // Convert BGR to RGB and normalize
            normalizedBuffer[i] = (Float32(dstData.load(fromByteOffset: i * 4 + 2, as: UInt8.self)) / 255.0 - mean[0]) / std[0] // R
            normalizedBuffer[width * height + i] = (Float32(dstData.load(fromByteOffset: i * 4 + 1, as: UInt8.self)) / 255.0 - mean[1]) / std[1] // G
            normalizedBuffer[width * height * 2 + i] = (Float32(dstData.load(fromByteOffset: i * 4 + 0, as: UInt8.self)) / 255.0 - mean[2]) / std[2] // B
        }
        
        free(dstData)
        return normalizedBuffer
    }
}

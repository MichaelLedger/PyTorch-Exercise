import Accelerate
import Foundation
import UIKit

/// Constants for the TeacherModel Core ML preprocessing
enum TeacherModelCoreMLConstants {
    /// Input dimensions required by the model
    static let inputWidth = 224
    static let inputHeight = 224
    
    /// ImageNet normalization parameters
    static let normalizationMean: [Float32] = [0.485, 0.456, 0.406]
    static let normalizationStd: [Float32] = [0.229, 0.224, 0.225]
    
    /// Image resampling method for resizing
    static let resamplingMethod: ResamplingMethod = .lanczos
    
    /// Available resampling methods
    enum ResamplingMethod {
        case nearest
        case bilinear
        case highQuality
        case lanczos
    }
}

extension CVPixelBuffer {
    /// Preprocesses the image for the TeacherModel Core ML:
    /// 1. Resizes to 224x224
    /// 2. Converts to RGB format
    /// 3. Scales pixel values to [0,1]
    /// 4. Applies ImageNet normalization
    ///
    /// - Returns: Normalized float array in CHW format (3x224x224) or nil if preprocessing fails
    func normalizedForTeacherModel() -> [Float32]? {
        let width = TeacherModelCoreMLConstants.inputWidth
        let height = TeacherModelCoreMLConstants.inputHeight
        let bytesPerRow = CVPixelBufferGetBytesPerRow(self)
        let bytesPerPixel = 4
        
        // Lock buffer for reading
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(self, .readOnly)
        }
        
        guard let baseAddr = CVPixelBufferGetBaseAddress(self) else {
            return nil
        }
        
        // Create vImage buffers for resizing
        var inBuff = vImage_Buffer(
            data: baseAddr,
            height: UInt(CVPixelBufferGetHeight(self)),
            width: UInt(CVPixelBufferGetWidth(self)),
            rowBytes: bytesPerRow
        )
        
        guard let dstData = malloc(width * height * bytesPerPixel) else {
            return nil
        }
        defer {
            free(dstData)
        }
        
        var outBuff = vImage_Buffer(
            data: dstData,
            height: UInt(height),
            width: UInt(width),
            rowBytes: width * bytesPerPixel
        )
        
        // Set resampling flags
        let flags: vImage_Flags
        switch TeacherModelCoreMLConstants.resamplingMethod {
        case .nearest:
            flags = vImage_Flags(kvImageNoFlags)
        case .bilinear:
            flags = vImage_Flags(kvImageNoFlags | kvImageHighQualityResampling)
        case .highQuality:
            flags = vImage_Flags(kvImageHighQualityResampling)
        case .lanczos:
            flags = vImage_Flags(kvImageHighQualityResampling | kvImageEdgeExtend)
        }
        
        // Perform resizing
        let error = vImageScale_ARGB8888(&inBuff, &outBuff, nil, flags)
        guard error == kvImageNoError else {
            return nil
        }
        
        // Create normalized buffer in CHW format (3x224x224)
        var normalizedBuffer = [Float32](repeating: 0, count: 3 * width * height)
        let mean = TeacherModelCoreMLConstants.normalizationMean
        let std = TeacherModelCoreMLConstants.normalizationStd
        
        // Process pixels in batches to reduce memory pressure
        let batchSize = 1024 * 16  // Process 16K pixels at a time
        for batchStart in stride(from: 0, to: width * height, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, width * height)
            
            autoreleasepool {
                for i in batchStart ..< batchEnd {
                    // Get RGB values (BGRA format)
                    let r = Float32(dstData.load(fromByteOffset: i * 4 + 2, as: UInt8.self))
                    let g = Float32(dstData.load(fromByteOffset: i * 4 + 1, as: UInt8.self))
                    let b = Float32(dstData.load(fromByteOffset: i * 4 + 0, as: UInt8.self))
                    
                    // Normalize and store in CHW format
                    normalizedBuffer[i] = (r / 255.0 - mean[0]) / std[0]                     // R channel
                    normalizedBuffer[width * height + i] = (g / 255.0 - mean[1]) / std[1]    // G channel
                    normalizedBuffer[width * height * 2 + i] = (b / 255.0 - mean[2]) / std[2] // B channel
                }
            }
        }
        
        return normalizedBuffer
    }
    
    /// Helper method to create a debug visualization of the preprocessed image
    func debugPreprocessedImage() -> UIImage? {
        guard let normalizedData = normalizedForTeacherModel() else {
            return nil
        }
        
        let width = TeacherModelCoreMLConstants.inputWidth
        let height = TeacherModelCoreMLConstants.inputHeight
        
        // Create RGB buffer for visualization
        var rgbData = [UInt8](repeating: 0, count: width * height * 3)
        
        // Denormalize values back to 0-255 range
        let mean = TeacherModelCoreMLConstants.normalizationMean
        let std = TeacherModelCoreMLConstants.normalizationStd
        
        for i in 0..<(width * height) {
            // Denormalize and clamp values
            let r = UInt8(max(0, min(255, (normalizedData[i] * std[0] + mean[0]) * 255.0)))
            let g = UInt8(max(0, min(255, (normalizedData[width * height + i] * std[1] + mean[1]) * 255.0)))
            let b = UInt8(max(0, min(255, (normalizedData[width * height * 2 + i] * std[2] + mean[2]) * 255.0)))
            
            rgbData[i * 3] = r
            rgbData[i * 3 + 1] = g
            rgbData[i * 3 + 2] = b
        }
        
        // Create CGImage from RGB data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let provider = CGDataProvider(data: Data(rgbData) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 24,
                bytesPerRow: width * 3,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
}
//
//  MUSIQPredictor.swift
//  PyTorchDemo
//
//  Created by Gavin Xiang on 2023/4/26.
//

import UIKit

class MUSIQPredictor: Predictor {
    private var isRunning: Bool = false
    private lazy var module: MUSIQTorchModule = {
        // resnet50_script_module.pt / resnet50_mobile_model.ptl
        // IQA_script_module.pt / IQA_mobile_model.ptl
        if let filePath = Bundle.main.path(forResource: "resnet50_mobile_model", ofType: "ptl"),
           let subFilePath = Bundle.main.path(forResource: "IQA_mobile_model", ofType: "ptl"),
           let module = MUSIQTorchModule(fileAtPath: filePath, subFilePath: subFilePath) {
            return module
        } else {
            fatalError("Failed to load model!")
        }
    }()

    private var labels: [String] = {
        if let filePath = Bundle.main.path(forResource: "words", ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Label file was not found.")
        }
    }()

//    func predict(_ buffer: [Float32], resultCount: Int) throws -> ([InferenceResult], Double)? {
//        if isRunning {
//            return nil
//        }
//        isRunning = true
//        let startTime = CACurrentMediaTime()
//        var tensorBuffer = buffer
//        guard let outputs = module.predict(image: UnsafeMutableRawPointer(&tensorBuffer)) else {
//            throw PredictorError.invalidInputTensor
//        }
//        isRunning = false
//        let inferenceTime = (CACurrentMediaTime() - startTime) * 1000
//        let results = topK(scores: outputs, labels: labels, count: resultCount)
//        return (results, inferenceTime)
//    }
    
    func predict(_ image: UIImage) throws -> (Float, Double)? {
        if isRunning {
            return nil
        }
        isRunning = true
        let startTime = CACurrentMediaTime()
        
//        guard let cgImgWidth = image.cgImage?.width, let cgImgHeight = image.cgImage?.height else {
//            return nil
//        }
        
        let scaledImageSize = CGSizeMake(CGFloat(MUSIQConstants.inputImageWidth), CGFloat(MUSIQConstants.inputImageHeight))
        guard let resizedImage = image.scaledImage(with: scaledImageSize) else {
            return nil
        }
        
        // original image
        guard let pixelBuffer = resizedImage.pixelBuffer() else {
            return nil
        }
//        let originalSize = CGSizeMake(CGFloat(cgImgWidth), CGFloat(cgImgHeight))
        guard let normalizedBuffer = pixelBuffer.normalized(Int(scaledImageSize.width), Int(scaledImageSize.height)) else {
            return nil
        }
        var tensorBuffer = normalizedBuffer
        
        // scaled image 1
//        let scaledImageHeight1: CGFloat = CGFloat(cgImgHeight) * (CGFloat(MUSIQConstants.inputImageWidth1) / CGFloat(cgImgWidth))
        let scaledImageSize1 = CGSizeMake(CGFloat(MUSIQConstants.inputImageWidth1), CGFloat(MUSIQConstants.inputImageHeight1))
        guard let resizedImage1 = image.scaledImage(with: scaledImageSize1) else {
            return nil
        }
//        guard let cropedToSquareImage = resizedImage1.cropToSquare() else {
//            return nil
//        }
        guard let pixelBuffer1 = resizedImage1.pixelBuffer() else {
            return nil
        }
        guard let normalizedBuffer1 = pixelBuffer1.normalized(Int(scaledImageSize1.width), Int(scaledImageSize1.height)) else {
            return nil
        }
        var tensorBuffer1 = normalizedBuffer1
        
        // scaled image 2
//        let scaledImageHeight2: CGFloat = CGFloat(cgImgHeight) * (CGFloat(MUSIQConstants.inputImageWidth2) / CGFloat(cgImgWidth))
        let scaledImageSize2 = CGSizeMake(CGFloat(MUSIQConstants.inputImageWidth2), CGFloat(MUSIQConstants.inputImageHeight2))
        guard let resizedImage2 = image.scaledImage(with: scaledImageSize2) else {
            return nil
        }
//        guard let cropedToSquareImage = resizedImage2.cropToSquare() else {
//            return nil
//        }
        guard let pixelBuffer2 = resizedImage2.pixelBuffer() else {
            return nil
        }
        guard let normalizedBuffer2 = pixelBuffer2.normalized(Int(scaledImageSize2.width), Int(scaledImageSize2.height)) else {
            return nil
        }
        var tensorBuffer2 = normalizedBuffer2
        
        let score = module.predict(image: UnsafeMutableRawPointer(&tensorBuffer), size: scaledImageSize, scaled1: UnsafeMutableRawPointer(&tensorBuffer1), size1: scaledImageSize1, scaled2: UnsafeMutableRawPointer(&tensorBuffer2), size2: scaledImageSize2)
        
        isRunning = false
//        let inferenceTime = (CACurrentMediaTime() - startTime) * 1000
        let inferenceTime = CACurrentMediaTime() - startTime
//        let results = topK(scores: outputs, labels: labels, count: resultCount)
//        guard let cgImage = postprocessImageData(numbers: outputs) else {
//            return (results, inferenceTime)
//        }
//        let outputImage = UIImage(cgImage: cgImage) //test
//        UIImageWriteToSavedPhotosAlbum(outputImage, nil, nil, nil)
//        print("[DEBUG] SavedPhotosAlbum:\(outputImage)")
        return (score, inferenceTime)
    }
    
    /*
    private func postprocessImageData(numbers: [NSNumber],
                                      size: CGSize = MUSIQConstants.inputImageSize) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        var floats: [Float32] = []
        numbers.forEach{ floats.append($0.floatValue > 1.0 ? 1.0 : $0.floatValue)} //test
//        let floats = data.toArray(type: Float32.self)
        
        let bufferCapacity = width * height * 4
        let unsafePointer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferCapacity)
        let unsafeBuffer = UnsafeMutableBufferPointer<UInt8>(start: unsafePointer,
                                                             count: bufferCapacity)
        defer {
            unsafePointer.deallocate()
        }
        // Fatal error: Float value cannot be converted to UInt8 because the result would be greater than UInt8.max
        for x in 0 ..< width {
            for y in 0 ..< height {
                let floatIndex = (y * width + x) * 3
                let index = (y * width + x) * 4
                let red = UInt8(floats[floatIndex] * 255)
                let green = UInt8(floats[floatIndex + 1] * 255)
                let blue = UInt8(floats[floatIndex + 2] * 255)
                
                unsafeBuffer[index] = red
                unsafeBuffer[index + 1] = green
                unsafeBuffer[index + 2] = blue
                unsafeBuffer[index + 3] = 0
            }
        }
        
        let outData = Data(buffer: unsafeBuffer)
        
        // Construct image from output tensor data
        let alphaInfo = CGImageAlphaInfo.noneSkipLast
        let bitmapInfo = CGBitmapInfo(rawValue: alphaInfo.rawValue)
            .union(.byteOrder32Big)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard
            let imageDataProvider = CGDataProvider(data: outData as CFData),
            let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: MemoryLayout<UInt8>.size * 4 * Int(MUSIQConstants.inputImageSize.width),
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: imageDataProvider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            )
        else {
            return nil
        }
        return cgImage
    }
     */
}
